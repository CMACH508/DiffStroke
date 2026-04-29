<div align="center">

# DiffStroke

### Harnessing Diffusion Models for Image Manipulation With Partial Sketches

**Official PyTorch implementation** of the IEEE Transactions on Image Processing paper  
*Harnessing Diffusion Models for Image Manipulation With Partial Sketches*

<p>
  <a href="https://doi.org/10.1109/TIP.2026.3685834"><img src="https://img.shields.io/badge/Paper-IEEE%20TIP-blue" alt="IEEE TIP"></a>
  <img src="https://img.shields.io/badge/Python-3.9-3776AB" alt="Python 3.9">
  <img src="https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C" alt="PyTorch 2.3.0">
  <img src="https://img.shields.io/badge/Backbone-Stable%20Diffusion%201.5-purple" alt="Stable Diffusion 1.5">
  <img src="https://img.shields.io/badge/Task-Mask--free%20Sketch%20Editing-green" alt="Mask-free sketch editing">
</p>

[Paper](https://doi.org/10.1109/TIP.2026.3685834) · [Checkpoints](#checkpoints) · [Datasets](#datasets) · [Quick Start](#quick-start) · [Citation](#citation)

</div>


**DiffStroke** is a mask-free framework for localized image manipulation with partial sketches. Given a source image, a few user strokes, and an optional text prompt, DiffStroke edits the intended local structure while preserving irrelevant regions, without requiring a manually drawn mask.

## Highlights

- **Mask-free local editing.** Users only provide partial strokes; the editing region is estimated automatically.
- **Precise sparse-stroke control.** Image-Stroke Fusion (ISF) blocks fuse source-image features and stroke features inside the diffusion backbone.
- **Region preservation.** A lightweight self-supervised mask estimator protects non-edited areas during denoising.
- **Text-sketch cooperation.** Text prompts can be combined with the same strokes to generate semantically diverse edits.
- **Natural and facial images.** The released code supports natural-scene editing and face editing checkpoints.

## Method Overview

<p align="center">
  <img src="assets/pipeline.png" width="95%" alt="DiffStroke training and inference pipeline">
</p>

DiffStroke builds on a pretrained sketch-controlled diffusion backbone. During training, Stable Diffusion and the T2I-Adapter sketch encoder are frozen; only the ISF blocks and the lightweight mask prediction branch are optimized.

<table>
<tr>
<td width="33%" align="center"><b>1. Feature-level fusion</b></td>
<td width="33%" align="center"><b>2. Self-supervised mask estimation</b></td>
<td width="33%" align="center"><b>3. DDIM inversion editing</b></td>
</tr>
<tr>
<td>ISF blocks bridge the mismatch between sparse partial sketches and dense diffusion feature maps.</td>
<td>Tweedie's formula is used to estimate clean latent targets and train the mask branch without manual region labels.</td>
<td>The source image is inverted into the diffusion trajectory and blended with the edited trajectory using the estimated mask.</td>
</tr>
</table>

## Results

### Natural Images

<p align="center">
  <img src="assets/natural_results.png" width="98%" alt="DiffStroke qualitative results on natural images">
</p>

### Face Editing

<p align="center">
  <img src="assets/face_results.png" width="98%" alt="DiffStroke qualitative results on face images">
</p>

## Repository Structure

```text
.
├── Dataset.py                         # Data loading, FFD deformation, sketch/mask construction
├── Inference.py                       # Inference on user-provided images and strokes
├── train_partial_sketch.py            # Natural-image training on Sketchy-style data
├── train_face.py                      # Face fine-tuning on CelebA-HQ
├── app_coadapter.py                   # CoAdapter demo code kept for reference
├── dist_util.py                       # Distributed utility functions
├── path_utils.py                      # Path defaults and legacy fallback support
├── configs/
│   └── stable-diffusion/train_sketch.yaml
├── ldm/                               # Diffusion model, sampler, adapter, and ISF modules
├── Custom_Dataset/                    # Example/custom-data helper directory
├── checkpoints/                       # Recommended checkpoint root, not tracked by git
├── data_train/                        # Recommended training-data root, not tracked by git
├── data_test/                         # Recommended testing-data root, not tracked by git
└── outputs/                           # Logs, checkpoints, masks, and visualizations
```

The code also supports the previous local layout (`models/`, `Dataset/`, and `experiments/`) through automatic fallback in `path_utils.py`. For a clean public setup, the recommended roots are `checkpoints/`, `data_train/`, `data_test/`, `data/`, and `outputs/`.

## Installation

The code was developed with Python 3.9 and CUDA GPUs.

```bash
conda create -n diffstroke python=3.9
conda activate diffstroke
pip install -r requirements.txt
conda install mkl
```

The provided environment uses PyTorch 2.3.0, torchvision 0.18.0, and xFormers 0.0.26.post1. If these wheels do not match your CUDA version, install a compatible PyTorch build first and then install the remaining dependencies.

If `basicsr` raises an import error with torchvision, modify:

```text
<conda-env>/lib/python3.9/site-packages/basicsr/data/degradations.py
```

and replace the grayscale import with:

```python
from torchvision.transforms._functional_tensor import rgb_to_grayscale
```

## Checkpoints

The easiest setup is to download and extract the complete `checkpoints/` archive:

- [Complete checkpoints archive](https://drive.google.com/file/d/15rzxYdyNpHj4amygk0d3IA7Cj4ND2kf4/view?usp=sharing), including the DiffStroke pretrained weights

The DiffStroke pretrained weights are also available separately:

- [DiffStroke pretrained weights](https://drive.google.com/file/d/1Z6XaPaP24RN-rzb7SrwyHYcyTseOysBz/view)

After extraction, arrange files as follows:

```text
checkpoints/
├── stable-diffusion-v1-5/
│   └── v1-5-pruned-emaonly.ckpt
├── clip-vit-large-patch14/
├── t2i-adapter/
│   └── t2iadapter_sketch_sd15v2.pth
├── pidinet/
│   └── table5_pidinet.pth
├── dlib/
│   └── shape_predictor_68_face_landmarks.dat
└── diffstroke/
    ├── natural/
    │   ├── fusionnet.pth
    │   └── model_fusion_170000.pth
    └── face/
        └── model_fusion_30000.pth
```

All checkpoint paths can be overridden from the command line:

```bash
python Inference.py \
  --ckpt checkpoints/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --adapter_ckpt checkpoints/t2i-adapter/t2iadapter_sketch_sd15v2.pth \
  --pidinet_ckpt checkpoints/pidinet/table5_pidinet.pth \
  --fusion_ckpt checkpoints/diffstroke/natural/model_fusion_170000.pth
```

The CLIP path in the Stable Diffusion config is resolved from `--checkpoint_root`; the recommended location is:

```text
checkpoints/clip-vit-large-patch14/
```

## Datasets

Download the prepared datasets from Google Drive:

- [Training dataset](https://drive.google.com/file/d/1NbA1soQ0FJb5WysYgQcyi-50X97CI71k/view?usp=sharing)
- [Testing dataset](https://drive.google.com/file/d/1V7V1bc2aZdO-EXAd1OMR5EYJqzjJHbKT/view?usp=sharing)

Extract the training and testing data to the dataset roots used by the commands below. The default examples use `data_train/` for training data and `data_test/` for testing data.

### Natural Images

For generic-scene training, the paper uses the Sketchy dataset with 11,250 training images. The code expects:

```text
data_train/sketchy/
├── info-06-04/
│   └── info/
│       └── testset.txt
└── rendered_256x256/
    └── 256x256/
        ├── sketch/
        │   └── tx_000000000000/
        │       └── <category>/
        │           └── <image-name>-<sketch-id>.png
        └── photo/
            ├── tx_000000000000/
            │   └── <category>/
            │       └── <image-name>.jpg
            └── caption/
                └── <category>/
                    └── <image-name>.txt
```

Training pairs are constructed automatically by free-form deformation (FFD), PiDiNet edge extraction, and deformation-region estimation.

### Face Images

For face editing, the paper uses CelebA-HQ with 28,000 training images and 2,000 testing images. The code expects:

```text
data_train/CelebA-HQ/
├── train_split.txt
├── test_split.txt
├── CelebA-HQ-img/
│   └── <image-id>.jpg
└── captions/
    └── <image-id>.text
```

Face deformation is generated using a mixture of FFD and landmark-based deformation. The landmark branch requires:

```text
checkpoints/dlib/shape_predictor_68_face_landmarks.dat
```

### Custom Images

For custom inference, arrange files as:

```text
data/examples/YourCase/
├── images/
│   ├── 0.png
│   └── 1.png
├── edges/
│   ├── 0.png
│   └── 1.png
└── captions/
    ├── 0.text
    └── 1.text
```

Each image, sketch, and caption must share the same numeric file name. The first line of each `.text` file is used as the text prompt.

## Quick Start

Run DiffStroke on custom samples:

```bash
python Inference.py \
  --data_path data/examples/YourCase \
  --result_dir outputs/user_results/YourCase \
  --ckpt checkpoints/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --adapter_ckpt checkpoints/t2i-adapter/t2iadapter_sketch_sd15v2.pth \
  --pidinet_ckpt checkpoints/pidinet/table5_pidinet.pth \
  --fusion_ckpt checkpoints/diffstroke/natural/model_fusion_170000.pth \
  --config configs/stable-diffusion/train_sketch.yaml \
  --n_samples 1 \
  --ddim_steps 50 \
  --scale 3.5
```

Outputs are saved under `--result_dir` when provided:

```text
outputs/user_results/YourCase/
├── visualization/          # Edited images
└── gen_mask/               # Estimated editing masks
```

## Training

### Natural Image Training

```bash
python train_partial_sketch.py \
  --data_path data_train/sketchy \
  --ckpt checkpoints/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --adapter_ckpt checkpoints/t2i-adapter/t2iadapter_sketch_sd15v2.pth \
  --pidinet_ckpt checkpoints/pidinet/table5_pidinet.pth \
  --config configs/stable-diffusion/train_sketch.yaml \
  --output_dir outputs \
  --bsize 4
```

The paper trains the natural-image model for 170,000 steps with AdamW, learning rate `1e-4`, batch size `4`, and loss weights `lambda_1 = 2.5`, `lambda_2 = 0.25`.

Checkpoints and logs are saved to:

```text
outputs/train_sketch/
├── models/
├── training_states/
└── visualization/
```

Resume from the latest training state:

```bash
python train_partial_sketch.py --output_dir outputs --auto_resume
```

### Face Fine-Tuning

```bash
python train_face.py \
  --data_path data_train/CelebA-HQ \
  --ckpt checkpoints/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --adapter_ckpt checkpoints/t2i-adapter/t2iadapter_sketch_sd15v2.pth \
  --pidinet_ckpt checkpoints/pidinet/table5_pidinet.pth \
  --init_fusion_ckpt checkpoints/diffstroke/natural/model_fusion_170000.pth \
  --landmark_model checkpoints/dlib/shape_predictor_68_face_landmarks.dat \
  --config configs/stable-diffusion/train_sketch.yaml \
  --output_dir outputs \
  --bsize 4
```

The face editing model is initialized from the natural-image DiffStroke checkpoint and fine-tuned for another 30,000 steps on CelebA-HQ.

## Evaluation

### Natural Images

```bash
python Inference.py \
  --data_path data_test/Places2 \
  --result_dir outputs/test_natural \
  --ckpt checkpoints/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --adapter_ckpt checkpoints/t2i-adapter/t2iadapter_sketch_sd15v2.pth \
  --pidinet_ckpt checkpoints/pidinet/table5_pidinet.pth \
  --fusion_ckpt checkpoints/diffstroke/natural/model_fusion_170000.pth \
  --config configs/stable-diffusion/train_sketch.yaml \
  --n_samples 1 \
  --ddim_steps 50 \
  --scale 3.5
```

### Face Images

```bash
python Inference.py \
  --data_path data_test/CelebA-HQ \
  --result_dir outputs/test_face \
  --ckpt checkpoints/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt \
  --adapter_ckpt checkpoints/t2i-adapter/t2iadapter_sketch_sd15v2.pth \
  --pidinet_ckpt checkpoints/pidinet/table5_pidinet.pth \
  --fusion_ckpt checkpoints/diffstroke/face/model_fusion_30000.pth \
  --config configs/stable-diffusion/train_sketch.yaml \
  --n_samples 1 \
  --ddim_steps 50 \
  --scale 3.5
```

## Important Arguments

| Argument | Purpose |
| --- | --- |
| `--data_path` | Dataset or custom example directory |
| `--checkpoint_root` | Root used to resolve CLIP and default checkpoints |
| `--ckpt` | Stable Diffusion v1.5 checkpoint |
| `--adapter_ckpt` | T2I-Adapter sketch checkpoint |
| `--pidinet_ckpt` | PiDiNet edge extractor checkpoint |
| `--fusion_ckpt` | DiffStroke checkpoint for evaluation or inference |
| `--init_fusion_ckpt` | Natural-image checkpoint used to initialize face fine-tuning |
| `--landmark_model` | dlib face landmark predictor |
| `--output_dir` | Training and evaluation output root |
| `--result_dir` | Custom inference output directory |

## Implementation Notes

- The default Stable Diffusion backbone is v1.5.
- The default sampler uses 50 DDIM steps.
- The feed-forward dimension in each ISF transformer block is 1024.
- The source feature used by ISF is extracted from a noisy source latent at timestep `t = 273`.
- The code is written for single-GPU CUDA execution. Some scripts initialize a distributed environment for compatibility.

## Citation

If you find this project useful, please cite:

```bibtex
@ARTICLE{11495049,
  author={Li, Tengjie and Tu, Shikui and Xu, Lei},
  journal={IEEE Transactions on Image Processing},
  title={Harnessing Diffusion Models for Image Manipulation With Partial Sketches},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Feeds;Antennas;Feedback;Circuits;Pixel;Videos;Video equipment;Communication systems;Text to video;Computer networks;sketch-based image manipulation;conditional controllable image synthesis;mask-free;diffusion models},
  doi={10.1109/TIP.2026.3685834}}
```

## Acknowledgements

This implementation builds on Stable Diffusion, T2I-Adapter/CoAdapter, PiDiNet, BasicSR, and related open-source projects. Please follow their licenses and model usage terms.

---
