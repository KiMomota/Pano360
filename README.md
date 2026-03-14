<div align="center">
<h1>Pano360: Perspective to Panoramic Vision with Geometric Consistency</h1>

<a href="https://arxiv.org/abs/2503.11651"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Pano360-blue)](https://huggingface.co/datasets/DongZhi/Pano360)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

🚀 The [Pano360 Dataset](https://huggingface.co/datasets/DongZhi/Pano360) contains four scenes (a, b, c, d), including tourism, sports, and special lighting scenes.

Here is a clean, professional, and well-formatted English version tailored specifically for a **GitHub Release** or **README.md** file. 

You can directly copy and paste the Markdown code below:

***

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

