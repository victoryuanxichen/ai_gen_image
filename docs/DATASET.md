# How the dataset is built

The dataset is assembled from public benchmarks by a registry file
(`configs/data.yaml`) and the `aidetect manifest` command. Nothing is copied or
converted: the manifest is a CSV with one row per image file, and every later
stage reads only the manifest.

## Sources

| Dataset | Real images | Generators | Used for | Where to get it |
| --- | --- | --- | --- | --- |
| GenImage (Zhu et al., NeurIPS 2023) | ImageNet-1K | Stable Diffusion v1.4 and v1.5, Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN | SD v1.4 for training and hold-out; the other seven for testing | https://github.com/GenImage-Dataset/GenImage (links to the download) |
| ForenSynths / CNNDetection (Wang et al., CVPR 2020) | LSUN and others | ProGAN (training set, 20 object classes); test set with StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN and others | ProGAN for training and hold-out; other GANs for testing | https://github.com/PeterWang512/CNNDetection |
| UniversalFakeDetect diffusion test set (Ojha et al., CVPR 2023) | 1k real per domain | LDM and GLIDE variants and others, 1k images each | testing only (optional) | https://github.com/WisconsinAIVision/UniversalFakeDetect |

Check each dataset's licence and terms before use, in particular for commercial
use. GenImage's real images are ImageNet images and inherit ImageNet's terms.

The requirement's reference list also names GigaGAN and DALL-E. The READMEs of
the three benchmarks above do not list either one (the UniversalFakeDetect
paper evaluates DALL-E images, so check whether your download includes them).
Any folder of generated images, with a folder of real images to pair it with,
can be added to the registry as an extra `unseen` source.

## Roles

| Role | Meaning | Default assignment |
| --- | --- | --- |
| `seen` | Used for training. Its held-out images form the hold-out test set. | SD v1.4 (diffusion), ProGAN (GAN) |
| `near_seen` | Never trained on, but the same architecture family as a seen generator. Reported separately. | SD v1.5, Wukong |
| `unseen` | Never trained on, different family or version. This is the generalization test. | Midjourney, ADM, GLIDE, VQDM, BigGAN, StyleGAN, StyleGAN2, CycleGAN, StarGAN, GauGAN |

SD v1.5 and Wukong are Stable Diffusion checkpoints. A detector trained on
SD v1.4 scores close to its hold-out result on them, so counting them as unseen
would flatter the generalization number.

## Splits

- **train / val.** Each `split: train` source is divided by a hash of the file
  path: 5% goes to `val` (set by `val_fraction`), the rest to `train`. The split
  is therefore stable across runs and machines. `val` is used for model
  selection, probability calibration and choosing the threshold.
- **test.** Each benchmark's own held-out folder (GenImage `val`, ForenSynths
  `test`). GenImage's `val` real images come from the ImageNet validation set,
  so they do not overlap the ImageNet training images used for training.

`max_per_label` caps how many real and how many generated images are taken from
a source (a deterministic subsample). The default of 60,000 per class for each
training source gives about 228,000 training and 12,000 validation images.

## Checks run by `aidetect manifest`

The command fails when:

- a test image is byte-identical to a training or validation image;
- a generator marked `near_seen` or `unseen` has images in training;
- a source with role `near_seen` or `unseen` is given a split other than `test`.

It warns when:

- real and generated images differ in file format (share of JPEG) by more than 20 points;
- real and generated images of one generator differ in median size by more than 1.5x;
- the content category could be read for fewer than half of a source's images.

The full audit is written next to the manifest as `<name>.audit.json`.

## Known biases and how they are handled

| Bias | Where it comes from | Handling |
| --- | --- | --- |
| File format | GenImage real images are JPEG (mostly quality 96); generated images are PNG. Detectors trained on the raw files learn "JPEG means real" (Grommelt et al., 2024). | Every non-JPEG input is re-encoded as JPEG quality 96 before the model sees it, in training, evaluation and serving. Training also re-compresses both classes at random qualities. |
| Image size | Each generator outputs one size (128 px for BigGAN up to 1024 px for Midjourney); ImageNet photos vary. | All inputs are resized to one size. Training applies random rescaling with random interpolation to both classes. The report breaks results down by resolution. |
| Content | ImageNet classes are object-centric; LSUN has 20 scene and object classes. People, text, art and screenshots are under-represented. | The report breaks results down by content category. Add a sample of your platform's own images as an extra source to measure what matters in production. |
| Real-image source | All real images are from ImageNet or LSUN, collected before generative models were common. | Not fixed by this project. A detector may behave differently on recent phone photos; test on your own real images. |
| Class balance | Sources differ in size. | Training samples real and generated 1:1 and spreads generated images evenly across generators. |

## Content categories for the bias analysis

Each row gets a `category` and a coarse `category_group`:

- **GenImage generated images**: the ImageNet class index is read from the
  start of the file name.
- **GenImage real training images**: the WordNet id is read from the file name
  and converted to a class index. This works automatically when all 1,000
  classes are present, or with ImageNet's `LOC_synset_mapping.txt`.
- **GenImage real test images** (ImageNet validation files): the file names
  carry no class, so set `imagenet_val_labels` to ImageNet's
  `LOC_val_solution.csv`. Without it these images have category `unknown` and
  the category analysis covers generated images only.
- **ForenSynths**: the class folder name (airplane, horse, ...).

ImageNet indices are grouped into seven coarse groups by index range: dog,
other animals, object, food, landscape, person, plants and fungi. The ranges are
approximate at the edges. Supply `category_groups_file` (a CSV of
`category,group`) for an exact taxonomy.

**The file-name patterns in `configs/data.yaml` are assumptions about the
benchmark's naming.** Confirm them on your copy: the manifest command reports
the share of images with a known category per source.

## Adding a generator later

Add a source to `configs/data.yaml` with `role: unseen` and `split: test`,
rebuild the manifest and re-run the evaluation. Cached scores for existing
images are reused, so only the new images are scored.
