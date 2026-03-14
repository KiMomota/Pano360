<div align="center">
  <h1>
    Pano360: Perspective to Panoramic Vision with Geometric Consistency
    <br><br>
    <!-- arXiv Badge -->
    <a href="https://arxiv.org/abs/2503.11651"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
    <!-- Hugging Face Badge -->
    <a href="https://huggingface.co/datasets/DongZhi/Pano360"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Pano360-blue" alt="Hugging Face Dataset"></a>
    <!-- License Badge -->
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  </h1>
</div>

🚀 The [Pano360 Dataset](https://huggingface.co/datasets/DongZhi/Pano360) contains four scenes (a, b, c, d), including tourism, sports, and special lighting scenes.

### 📂 Dataset Directory Structure

The dataset is organized hierarchically by scene categories, sub-scenes, and focal lengths. Below is the detailed layout:

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

