# AI-generated image detector

Classifies a JPEG or PNG image as **Real** or **AI-Generated** and returns a
calibrated probability. The project contains the training code, the evaluation
suite for the four acceptance criteria, and an inference module with an HTTP API.

**Status: code complete, no trained model yet.** The data, perturbation,
metric, calibration, evaluation-report and inference code is covered by tests
that run on CPU. The deep-model code (`torch_model.py`, `train.py`, `serve.py`)
was written without access to PyTorch or a GPU and has not been executed; run
`pytest` on your machine first (see step 1). Trained weights and the final
evaluation report are produced by steps 3 to 5 on your GPU.

## How it works

- **Model.** A frozen vision foundation model from `timm` (default DINOv3
  ViT-L/16) with a trained linear head. Only the head is trained, which fits a
  single consumer GPU.
- **Data.** Public benchmarks. The detector trains on one diffusion generator
  (Stable Diffusion v1.4, from GenImage) and one GAN (ProGAN, from ForenSynths).
  Every other generator is kept for the unseen-generator test.
- **Shortcut control.** Benchmarks store real photos as JPEG and generated
  images as PNG at fixed sizes. Every non-JPEG input is re-encoded as JPEG
  quality 96 and all inputs are brought to one size, in training, evaluation and
  serving alike, so the model cannot learn the file format or image size.
- **Robustness.** Random resizing, blur, noise and JPEG re-compression are
  applied to both classes during training.
- **One scoring path.** Validation, evaluation and the API all go through
  `aidetect/scoring.py` and `aidetect/inference.py`.

The design rationale is in the technical design document that accompanies this
repository. Dataset construction is documented in [docs/DATASET.md](docs/DATASET.md).

## Layout

```
aidetect/
  imageio.py      decoding, EXIF/alpha handling, format harmonization
  perturb.py      perturbations for robustness tests and training augmentation
  categories.py   content categories for the bias analysis
  manifest.py     dataset manifest builder and audit (leakage, shortcuts)
  metrics.py      precision, recall, F1, AUC, calibration error, confidence intervals
  calibrate.py    probability calibration and threshold selection
  scoring.py      shared scoring path and model bundles
  baseline.py     CPU spectral baseline (pipeline check, shortcut check)
  torch_model.py  timm backbone + head, bundle save/load        [needs PyTorch]
  train.py        training loop                                 [needs PyTorch]
  evaluate.py     the four acceptance evaluations
  report.py       Markdown evaluation report
  inference.py    ImageDetector: the inference module
  serve.py        FastAPI service                               [needs FastAPI]
  cli.py          command line
configs/          data registry, training and evaluation settings
scripts/          make_toy_data.py (synthetic data for smoke tests)
tests/            pytest suite
docs/DATASET.md   how the dataset is built
```

## Steps

### 1. Install and test

```bash
python -m venv .venv && source .venv/bin/activate
# install the PyTorch build for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pytest
```

`tests/test_deep.py` trains a tiny random ViT for one epoch on synthetic images
and exercises the HTTP API. It needs no downloads and runs on CPU.

Optional smoke test of the whole pipeline without a GPU:

```bash
python scripts/make_toy_data.py toy_data
python -m aidetect manifest --registry toy_data/registry.yaml --out toy_data/manifest.csv
python -m aidetect train-baseline --manifest toy_data/manifest.csv --out toy_data/run
python -m aidetect evaluate --manifest toy_data/manifest.csv --model toy_data/run/baseline.joblib \
    --out toy_data/eval --no-fail
```

The toy baseline is expected to fail the unseen and robustness checks. It only
shows that every stage runs and what the report looks like.

### 2. Get the data and build the manifest

Download the datasets listed in [docs/DATASET.md](docs/DATASET.md), set `root`
and check the folder paths in `configs/data.yaml`, then:

```bash
python -m aidetect manifest --registry configs/data.yaml --out data/manifest.csv
```

The command prints image counts per split, role and generator, and stops with
an error if a test image also appears in training or an unseen generator leaks
into training. Read its warnings: they flag format and size differences between
the classes and categories it could not read from file names.

### 3. Train

```bash
python -m aidetect train --manifest data/manifest.csv --config configs/train.yaml --out runs/main
```

Writes `runs/main/model.pt` (weights, preprocessing settings, calibration and
decision threshold in one file), `history.json` and `train_config.json`.
The default backbone's weights are under Meta's DINOv3 License and may require
accepting the terms on Hugging Face and `huggingface-cli login`. `configs/train.yaml`
lists alternatives.

### 4. Check for shortcuts (recommended)

```bash
python -m aidetect train-baseline --manifest data/manifest.csv --out runs/baseline
python -m aidetect evaluate --manifest data/manifest.csv --model runs/baseline/baseline.joblib \
    --out runs/baseline/eval --no-fail
```

If this trivial frequency model scores near the deep model on some test set,
that set has a low-level shortcut and its numbers overstate real ability.

### 5. Evaluate

```bash
python -m aidetect evaluate --manifest data/manifest.csv --model runs/main/model.pt \
    --config configs/eval.yaml --out runs/main/eval
```

Writes `REPORT.md`, `results.json` and per-image scores under `predictions/`.
The report has one section per acceptance criterion:

1. Hold-out test set: precision, recall, F1 and AUC with 95% confidence intervals.
2. Generators not used in training: per generator, macro average, drop from hold-out.
3. Robustness: 19 perturbation conditions with the change from clean.
4. Bias: results by content category, resolution and source format, with gaps.

The exit code is 2 when an acceptance check fails. The thresholds in
`configs/eval.yaml` are proposals; replace them with the agreed values.
Interrupted runs resume from the cached scores.

### 6. Use the model

Python:

```python
from aidetect.inference import ImageDetector

detector = ImageDetector.load("runs/main/model.pt")
result = detector.predict("photo.jpg")          # a path or raw bytes
print(result.label, result.confidence, result.ai_probability)
```

Command line:

```bash
python -m aidetect predict --model runs/main/model.pt photo.jpg other.png
```

HTTP service:

```bash
python -m aidetect serve --model runs/main/model.pt --port 8000
curl -F "file=@photo.jpg" http://localhost:8000/v1/detect
```

```json
{"label": "AI-Generated", "confidence": 0.97, "ai_probability": 0.97, "threshold": 0.5, "model": "..."}
```

(The values above illustrate the format only.)

| Field | Meaning |
| --- | --- |
| `label` | `"Real"` or `"AI-Generated"` |
| `confidence` | Probability assigned to the returned label, 0 to 1 |
| `ai_probability` | Calibrated probability that the image is AI-generated, 0 to 1 |
| `threshold` | `ai_probability` at or above this value gives `"AI-Generated"` |

Errors: 400 for undecodable or empty input, 413 above 20 MB
(`AIDETECT_MAX_BYTES`), 415 for formats other than JPEG and PNG. `GET /healthz`
reports readiness. Interactive API docs are served at `/docs`.

Downstream systems that need a different trade-off (for example very few false
alarms on real photos) should apply their own cut-off to `ai_probability`, or
train with `decision.target_fpr` set.

## Known limits

- A detector's score is evidence, not proof. Send borderline scores to human review.
- Published results for this kind of model show weak spots this project does
  not fix: images that were only partly edited by AI, real photos passed through
  a diffusion model's autoencoder, and screenshots or re-photographed images.
- New generators appear constantly. Add each new one to the registry as an
  `unseen` source, re-run step 5, and retrain when the unseen numbers drop.
