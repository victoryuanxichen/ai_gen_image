"""Train the deep detector from a manifest. Needs PyTorch, torchvision and timm."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import __version__, perturb
from .calibrate import Calibrator, threshold_for_fpr
from .evaluate import _sample, merge_config
from .imageio import ImageError, harmonize, open_image
from .manifest import balanced_weights, read_manifest
from .scoring import score_paths
from .torch_model import Detector, TorchScorer, save_bundle, train_transform

DEFAULT_TRAIN_CONFIG: dict = {
    "seed": 0,
    "pretrained": True,                  # False only for offline smoke tests
    "model": {
        "backbone": "vit_large_patch16_dinov3.lvd1689m",
        "head": "linear",
        "hidden_dim": 512,
        "dropout": 0.0,
        "unfreeze_last_n": 0,
        "img_size": None,
    },
    "data": {
        "image_size": None,              # None = the backbone's native input size
        "harmonize_jpeg_quality": 96,    # None disables format harmonization
        "crop_scale": [0.3, 1.0],
        "workers": 8,
    },
    "augment": {"enabled": True, **perturb.DEFAULT_AUGMENT},
    "train": {
        "epochs": 3,
        "batch_size": 128,
        "samples_per_epoch": None,       # None = one pass over the training rows
        "lr": 1.0e-3,
        "backbone_lr_mult": 0.01,
        "weight_decay": 1.0e-4,
        "warmup_fraction": 0.05,
        "label_smoothing": 0.0,
        "amp": True,
        "log_every": 50,
    },
    "validation": {
        "max_per_label": 5000,
        "perturbation": "resize:scale=0.5+jpeg:q=75",
    },
    "decision": {"threshold": 0.5, "target_fpr": None},
    "save_backbone": True,
}


class ManifestDataset(Dataset):
    """Training images: decode -> harmonize -> random degradation -> crop/flip."""

    def __init__(self, df: pd.DataFrame, transform, harmonize_q, augment: dict | None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy(dtype=np.float32)
        self.transform = transform
        self.harmonize_q = harmonize_q
        self.augment = augment
        self._rng = None
        self._rng_pid = None

    def __len__(self) -> int:
        return len(self.paths)

    def _generator(self) -> np.random.Generator:
        # One independent generator per DataLoader worker process.
        if self._rng is None or self._rng_pid != os.getpid():
            self._rng = np.random.default_rng()
            self._rng_pid = os.getpid()
        return self._rng

    def __getitem__(self, index: int):
        for _ in range(10):  # skip over the occasional corrupt file
            try:
                im, fmt = open_image(self.paths[index])
                break
            except ImageError:
                index = (index + 1) % len(self.paths)
        else:
            raise RuntimeError("10 consecutive unreadable images; check the dataset")
        im = harmonize(im, fmt, self.harmonize_q)
        if self.augment is not None:
            im = perturb.random_degrade(im, self._generator(), self.augment)
        return self.transform(im), self.labels[index]


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    keep = ~np.isnan(scores)
    if keep.sum() == 0 or len(np.unique(y[keep])) < 2:
        return float("nan")
    return float(roc_auc_score(y[keep], scores[keep]))


def train(config: dict | None, manifest_path: str, out_dir: str, device: str | None = None) -> Path:
    cfg = merge_config(DEFAULT_TRAIN_CONFIG, config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2))
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    df = read_manifest(manifest_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    if train_df["label"].nunique() < 2:
        raise ValueError("the train split must contain both real and generated images")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type != "cuda":
        print("WARNING: no GPU found; training a large backbone on CPU will be very slow")

    model_cfg = dict(cfg["model"])
    model = Detector(**model_cfg, pretrained=bool(cfg["pretrained"])).to(dev)
    data_cfg = model.data_config()
    if cfg["data"].get("image_size"):
        data_cfg["image_size"] = int(cfg["data"]["image_size"])
    elif model_cfg.get("img_size"):
        data_cfg["image_size"] = int(model_cfg["img_size"])
    harmonize_q = cfg["data"]["harmonize_jpeg_quality"]

    augment = dict(cfg["augment"])
    augment_enabled = augment.pop("enabled", True)
    dataset = ManifestDataset(
        train_df,
        train_transform(data_cfg["image_size"], data_cfg["mean"], data_cfg["std"], cfg["data"]["crop_scale"]),
        harmonize_q,
        augment if augment_enabled else None,
    )
    t = cfg["train"]
    batch_size = int(t["batch_size"])
    samples_per_epoch = int(t["samples_per_epoch"] or len(train_df))
    sampler = WeightedRandomSampler(
        torch.as_tensor(balanced_weights(train_df), dtype=torch.double),
        num_samples=samples_per_epoch,
        replacement=True,
    )
    workers = int(cfg["data"]["workers"])
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
        pin_memory=dev.type == "cuda", drop_last=True, persistent_workers=workers > 0,
    )
    if len(loader) == 0:
        raise ValueError("fewer training samples per epoch than one batch; lower batch_size")

    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": float(t["lr"])}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(t["lr"]) * float(t["backbone_lr_mult"])})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(t["weight_decay"]))
    total_steps = int(t["epochs"]) * len(loader)
    warmup = max(1, int(float(t["warmup_fraction"]) * total_steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    use_amp = bool(t["amp"]) and dev.type == "cuda"
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss()
    smoothing = float(t["label_smoothing"])

    scorer = TorchScorer(model, data_cfg, dev, amp=use_amp, name=model_cfg["backbone"])
    val_sample = _sample(val_df, int(cfg["validation"]["max_per_label"]), int(cfg["seed"])) if len(val_df) else val_df
    val_paths = list(val_sample["path"])
    val_y = val_sample["label"].to_numpy() if len(val_sample) else np.array([])
    val_spec = cfg["validation"]["perturbation"]

    def validate() -> dict:
        if not val_paths:
            return {"auc_clean": float("nan"), "auc_perturbed": float("nan"), "score": float("nan")}
        clean = score_paths(val_paths, scorer, harmonize_q, perturb.CLEAN, batch_size, workers)
        pert = score_paths(val_paths, scorer, harmonize_q, val_spec, batch_size, workers)
        a, b = _auc(val_y, clean), _auc(val_y, pert)
        return {"auc_clean": a, "auc_perturbed": b, "score": float(np.nanmean([a, b]))}

    history, best_score, best_state = [], -math.inf, None
    step = 0
    for epoch in range(int(t["epochs"])):
        model.train()
        running, seen, started = 0.0, 0, time.time()
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            if smoothing > 0:
                y = y * (1 - smoothing) + 0.5 * smoothing
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running += loss.item() * len(y)
            seen += len(y)
            if step % int(t["log_every"]) == 0:
                rate = seen / max(time.time() - started, 1e-6)
                print(f"epoch {epoch + 1} step {step}/{total_steps} loss {running / seen:.4f} {rate:.0f} img/s")
        val = validate()
        record = {"epoch": epoch + 1, "train_loss": running / max(seen, 1), **val}
        history.append(record)
        print(json.dumps(record))
        (out / "history.json").write_text(json.dumps(history, indent=2))
        score = val["score"]
        if math.isnan(score) or score > best_score:
            best_score = score if not math.isnan(score) else best_score
            best_state = model.trainable_state()
            torch.save(best_state, out / "best_trainable.pt")

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # Calibrate and choose the threshold on validation data only.
    calibrator, threshold = Calibrator(), float(cfg["decision"]["threshold"])
    if val_paths:
        val_logits = score_paths(val_paths, scorer, harmonize_q, perturb.CLEAN, batch_size, workers)
        calibrator = Calibrator.fit(val_logits, val_y)
        target_fpr = cfg["decision"].get("target_fpr")
        if target_fpr:
            threshold = threshold_for_fpr(calibrator(val_logits[val_y == 0]), float(target_fpr))
    else:
        print("WARNING: no validation rows; probabilities are uncalibrated")

    bundle_path = out / "model.pt"
    save_bundle(
        bundle_path, model, model_cfg, data_cfg, calibrator.to_dict(), threshold, harmonize_q,
        meta={
            "model_name": f"{model_cfg['backbone']}+{model_cfg['head']}",
            "aidetect_version": __version__,
            "trained_on_generators": sorted(train_df.loc[train_df["label"] == 1, "generator"].unique().tolist()),
            "n_train": int(len(train_df)),
            "best_val_score": None if best_score == -math.inf else float(best_score),
        },
        save_backbone=bool(cfg["save_backbone"]),
    )
    print(f"saved {bundle_path} (threshold {threshold:.4f}, calibration {calibrator.to_dict()})")
    return bundle_path
