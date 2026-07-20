<div align="center">
<h1 align="center">
Pano360: Perspective to Panoramic Vision with Geometric Consistency
</h1>

<a href="https://arxiv.org/abs/2603.12013"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Pano360-blue)](https://huggingface.co/datasets/DongZhi/Pano360)
![alt text](https://img.shields.io/badge/License-MIT-green.svg)

</div>

<p align="center">
  <img src="result/teaser.jpg" width="80%">
</p>

## Requirements

- Python 3.9
- PyTorch 2.3
- GPU: CUDA-compatible GPU with ≥12GB VRAM
- The VGGT-Omega checkpoint at `model/vggt_omega_1b_512.pt`

## Installation

```bash
conda create -n Pano360 python=3.9
conda activate Pano360    
pip install -r requirements.txt
```

Install the optional LightGlue dependency for bundle adjustment:
```bash
pip install -r requirements-ba.txt
```

## Pretrained models
We use VGGT-Omega as the geometry backbone. Please download the pretrained checkpoint from the official repository: [VGGT-Omega](https://github.com/facebookresearch/vggt-omega)

The checkpoint is approximately 4.3 GiB. Then place the checkpoint as: model/vggt_omega_1b_512.pt

## Quick Start

Place overlapping images in one directory. Files are loaded in case-insensitive
filename order.

Run the standard pipeline without bundle adjustment:

```bash
python demo_stitch.py \
  --image-folder ./example/night \
  --output-path ./result/night_normal.jpg \
  --device cuda \
  --projection plane
```

Run LightGlue matching and bundle adjustment:

```bash
python demo_stitch_ba.py \
  --image-folder ./example/night \
  --output-path ./result/night_normal_ba.jpg \
  --device cuda \
  --projection plane \
  --extractor aliked
```

Use `python demo_stitch.py --help` or `python demo_stitch_ba.py --help` for the
complete option list.

## Projections and Views

| Type | Options |
| --- | --- |
| Projection | `auto`, `plane`, `cylindrical`, `spherical`, `mercator`, `panini`, `erp` |
| View | `normal`, `little_planet`, `rabbit_hole`, `fisheye`, `cubemap` |
| Seam | `torch_dp`, `torch_soft`, `no` |
| Blend | `multiband`, `feather`, `no` |

Generate a fixed 2:1 equirectangular panorama:

```bash
python demo_stitch.py \
  --image-folder ./example/littleplane \
  --output-path ./result/littleplane_erp.jpg \
  --device cuda \
  --projection erp \
  --erp-width 8192
```

Generate a Little Planet view:

```bash
python demo_stitch.py \
  --image-folder ./example/littleplane \
  --output-path ./result/littleplane_little_planet.jpg \
  --device cuda \
  --projection erp \
  --view little_planet \
  --view-size 2048 \
  --view-zoom 0.65
```

## Pipeline

```text
Images -> VGGT-Omega cameras -> optional LightGlue + BA
       -> CUDA projection -> GPU seam -> exposure compensation
       -> multi-band blending -> optional display view -> RGB output
```

## 📂 Dataset Directory Structure

🚀 The [Pano360](https://huggingface.co/datasets/DongZhi/Pano360) dataset contains four scenes (a, b, c, d), including tourism, sports, and special lighting scenes.

```text
ROOT/
├── Scene(a)/                  # Tourism scenes
│   ├── 0/                     # 1st sub-scene
│   │   ├── 001/               # 1st focal length
│   │   │   ├── cameras.json   # Ground truth camera parameters
│   │   │   └── images/        # Contains exactly 24 image frames
│   │   ├── 002/               # 2nd focal length
│   │   └── 003/               # 3rd focal length
│   ├── 1/                     # 2nd sub-scene
│   ├── ...                    # (Intermediate sub-scenes)
│   └── 165/                   # 166th sub-scene (Indexed 0 to 165)
│
├── Scene(b)/                  # Sports scenes
│   └── ...                   
│
├── Scene(c)/                  # Special lighting scenes
│   └── ...                    
│
└── Scene(d)/                  # Unsupervised in-the-wild scenes
    └── ...                    
```

## Citation

If you find Pano360 useful for your research, please cite:

```bibtex
@inproceedings{zhu2026pano360,
  title={Pano360: Perspective to Panoramic Vision with Geometric Consistency},
  author={Zhu, Zhengdong and Xue, Weiyi and Yang, Zuyuan and Zhou, Wenlve and Zhou, Zhiheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7600--7609},
  year={2026}
}
```
