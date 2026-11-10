"""The deep detector: a pre-trained backbone from `timm` plus a small head.

Needs PyTorch, torchvision and timm. Nothing else in the package imports this
module at import time, so the data, evaluation and report code work without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.data import resolve_model_data_config
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class Detector(nn.Module):
    """Backbone features -> one logit (higher = more likely AI-generated).

    By default the backbone is frozen and only the head is trained. That is the
    configuration reported to generalize best to unseen generators (Zhou et al.,
    2026) and it is cheap enough for a single consumer GPU. `unfreeze_last_n`
    fine-tunes the last transformer blocks as an experiment.
    """

    def __init__(
        self,
        backbone: str = "vit_large_patch16_dinov3.lvd1689m",
        head: str = "linear",
        hidden_dim: int = 512,
        dropout: float = 0.0,
        unfreeze_last_n: int = 0,
        img_size: int | None = None,
        pretrained: bool = True,
    ):
        super().__init__()
        kwargs = {"pretrained": pretrained, "num_classes": 0}
        if img_size:
            kwargs["img_size"] = int(img_size)  # ViT-style models only
        self.backbone = timm.create_model(backbone, **kwargs)
        dim = self.backbone.num_features
        if head == "linear":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1))
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
            )
        else:
            raise ValueError(f"head must be 'linear' or 'mlp', got '{head}'")

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone_frozen = unfreeze_last_n <= 0
        if not self.backbone_frozen:
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None:
                raise ValueError(
                    f"unfreeze_last_n needs a backbone with a .blocks list; '{backbone}' has none"
                )
            for block in list(blocks)[-int(unfreeze_last_n):]:
                for p in block.parameters():
                    p.requires_grad = True
            for name in ("norm", "fc_norm"):
                module = getattr(self.backbone, name, None)
                if isinstance(module, nn.Module):
                    for p in module.parameters():
                        p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()  # keep dropout / stochastic depth off when frozen
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_frozen:
            with torch.no_grad():
                feats = self.backbone(x)
        else:
            feats = self.backbone(x)
        return self.head(feats).squeeze(-1)

    def data_config(self) -> dict:
        cfg = resolve_model_data_config(self.backbone)
        return {
            "image_size": int(cfg["input_size"][-1]),
            "mean": [float(v) for v in cfg["mean"]],
            "std": [float(v) for v in cfg["std"]],
        }

    def trainable_state(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}


def eval_transform(image_size: int, mean, std):
    """Resize the short side to the model's input size, then centre crop.

    Bringing every image to one size removes the image-size shortcut present in
    public benchmarks, where each generator outputs one fixed resolution.
    """
    return T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def train_transform(image_size: int, mean, std, crop_scale=(0.3, 1.0)):
    return T.Compose([
        T.RandomResizedCrop(
            image_size, scale=tuple(crop_scale), ratio=(3 / 4, 4 / 3),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class TorchScorer:
    """Adapts a `Detector` to the scorer interface used by evaluation and serving."""

    def __init__(self, model: Detector, data_cfg: dict, device: torch.device | str,
                 amp: bool = True, name: str = "timm_detector"):
        self.model = model
        self.device = torch.device(device)
        self.transform = eval_transform(data_cfg["image_size"], data_cfg["mean"], data_cfg["std"])
        self.amp = bool(amp) and self.device.type == "cuda"
        self.name = name

    @torch.inference_mode()
    def score(self, images: list[Image.Image]) -> np.ndarray:
        was_training = self.model.training
        self.model.eval()
        try:
            x = torch.stack([self.transform(im) for im in images]).to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(x)
            return logits.float().cpu().numpy()
        finally:
            if was_training:
                self.model.train()

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> tuple["TorchScorer", dict]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("kind") != "torch_timm":
            raise ValueError(f"{path} is not a torch_timm bundle")
        full = bool(payload.get("full_state", False))
        # With a full state dict the file is self-contained; otherwise the frozen
        # backbone weights are fetched again from the timm hub.
        model = Detector(**payload["model_cfg"], pretrained=not full)
        result = model.load_state_dict(payload["state_dict"], strict=full)
        if not full:
            missing_trained = [k for k in result.missing_keys if k.startswith("head.")]
            if missing_trained or result.unexpected_keys:
                raise ValueError(
                    f"{path}: bundle does not match the model "
                    f"(missing {missing_trained}, unexpected {list(result.unexpected_keys)})"
                )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        name = payload.get("meta", {}).get("model_name", payload["model_cfg"].get("backbone", "detector"))
        scorer = cls(model, payload["data"], device, name=name)
        payload = {k: v for k, v in payload.items() if k != "state_dict"}
        return scorer, payload


def save_bundle(
    path: str | Path,
    model: Detector,
    model_cfg: dict,
    data_cfg: dict,
    calibration: dict,
    threshold: float,
    harmonize_q: int | None,
    meta: dict,
    save_backbone: bool = True,
) -> None:
    """Write the model with everything inference needs.

    The payload holds only tensors and plain Python values, so it loads with
    ``torch.load(..., weights_only=True)`` and cannot execute code on load.
    """
    state = (
        {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if save_backbone else model.trainable_state()
    )
    torch.save(
        {
            "kind": "torch_timm",
            "model_cfg": dict(model_cfg),
            "data": dict(data_cfg),
            "state_dict": state,
            "full_state": bool(save_backbone),
            "calibration": dict(calibration),
            "threshold": float(threshold),
            "harmonize_q": harmonize_q,
            "meta": dict(meta),
        },
        path,
    )
