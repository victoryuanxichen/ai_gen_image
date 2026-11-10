"""Command line entry point: ``python -m aidetect <command>``.

  manifest        scan dataset folders into a manifest CSV and audit it
  train           train the deep detector (needs a GPU and PyTorch)
  train-baseline  train the small CPU baseline (pipeline check / shortcut check)
  evaluate        run the four acceptance evaluations and write REPORT.md
  predict         classify image files
  serve           start the HTTP API
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def cmd_manifest(args) -> int:
    from .manifest import build_manifest, load_registry

    df, audit = build_manifest(load_registry(args.registry), probe=not args.no_probe, workers=args.workers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))

    summary = (
        df.assign(kind=df["label"].map({0: "real", 1: "generated"}))
        .groupby(["split", "role", "generator", "kind"]).size().unstack("kind", fill_value=0)
    )
    print(summary.to_string())
    print(f"\n{len(df)} images -> {out}")
    for message in audit["warnings"]:
        print(f"WARNING: {message}")
    for message in audit["errors"]:
        print(f"ERROR: {message}")
    if audit["errors"] and not args.allow_errors:
        print("Manifest has errors; fix the registry or pass --allow-errors to proceed anyway.")
        return 1
    return 0


def cmd_train(args) -> int:
    from .train import train

    train(_load_yaml(args.config), args.manifest, args.out, device=args.device)
    return 0


def cmd_train_baseline(args) -> int:
    from .baseline import train_baseline
    from .manifest import read_manifest

    path = train_baseline(
        read_manifest(args.manifest), Path(args.out) / "baseline.joblib",
        max_per_label=args.max_per_label, target_fpr=args.target_fpr, workers=args.workers,
    )
    print(f"saved {path}")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import run_evaluation
    from .manifest import read_manifest
    from .scoring import load_bundle

    results = run_evaluation(
        read_manifest(args.manifest),
        load_bundle(args.model, device=args.device),
        _load_yaml(args.config),
        args.out,
        progress=not args.quiet,
    )
    print(f"verdict: {results['verdict']}")
    for c in results["acceptance"]:
        status = {True: "pass", False: "FAIL", None: "n/a"}[c["passed"]]
        value = "n/a" if c["value"] is None or c["value"] != c["value"] else f"{c['value']:.4f}"
        print(f"  [{status:>4}] {c['check']}: {value} (target {c['op']} {c['limit']})")
    print(f"report: {Path(args.out) / 'REPORT.md'}")
    return 0 if results["verdict"] != "FAIL" or args.no_fail else 2


def cmd_predict(args) -> int:
    from .imageio import ImageError
    from .inference import ImageDetector

    detector = ImageDetector.load(args.model, device=args.device)
    status = 0
    for path in args.images:
        try:
            print(json.dumps({"path": path, **detector.predict(path).to_dict()}))
        except ImageError as exc:
            print(json.dumps({"path": path, "error": str(exc)}))
            status = 1
    return status


def cmd_serve(args) -> int:
    import os

    import uvicorn

    os.environ["AIDETECT_MODEL"] = args.model
    if args.device:
        os.environ["AIDETECT_DEVICE"] = args.device
    uvicorn.run("aidetect.serve:app", host=args.host, port=args.port, workers=1)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aidetect", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="build and audit the dataset manifest")
    p.add_argument("--registry", required=True, help="data registry YAML")
    p.add_argument("--out", required=True, help="output manifest CSV")
    p.add_argument("--no-probe", action="store_true", help="skip reading image headers (faster, no size/format audit)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--allow-errors", action="store_true")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("train", help="train the deep detector")
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", help="training config YAML (defaults are used for missing keys)")
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--device")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("train-baseline", help="train the CPU spectral baseline")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--max-per-label", type=int, default=5000)
    p.add_argument("--target-fpr", type=float)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_train_baseline)

    p = sub.add_parser("evaluate", help="run the acceptance evaluations")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True, help="model bundle (.pt or .joblib)")
    p.add_argument("--config", help="evaluation config YAML")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--device")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-fail", action="store_true", help="exit 0 even when acceptance checks fail")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="classify image files")
    p.add_argument("--model", required=True)
    p.add_argument("--device")
    p.add_argument("images", nargs="+")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("serve", help="start the HTTP API")
    p.add_argument("--model", required=True)
    p.add_argument("--device")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
