<div align="center">
<h1>Pano360: Perspective to Panoramic Vision with Geometric Consistency</h1>

<a href="https://arxiv.org/abs/2503.11651"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Pano360-blue)](https://huggingface.co/datasets/DongZhi/Pano360)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

🚀 The [Pano360 Dataset](https://huggingface.co/datasets/DongZhi/Pano360) contains four scenes (a, b, c, d), including tourism, sports, and special lighting scenes.

###  Data directory structure
```
ROOT
|
--- ENV_NAME_0                             # environment folder
|       |
|       ---- Easy                          # difficulty level
|       |      |
|       |      ---- P000                   # trajectory folder
|       |      |      |
|       |      |      +--- depth_left      # 000000_left_depth.npy - 000xxx_left_depth.npy
|       |      |      +--- depth_right     # 000000_right_depth.npy - 000xxx_right_depth.npy
|       |      |      +--- flow            # 000000_000001_flow/mask.npy - 000xxx_000xxx_flow/mask.npy
|       |      |      +--- image_left      # 000000_left.png - 000xxx_left.png 
|       |      |      +--- image_right     # 000000_right.png - 000xxx_right.png 
|       |      |      +--- seg_left        # 000000_left_seg.npy - 000xxx_left_seg.npy
|       |      |      +--- seg_right       # 000000_right_seg.npy - 000xxx_right_seg.npy
|       |      |      ---- pose_left.txt 
|       |      |      ---- pose_right.txt
|       |      |  
|       |      +--- P001
|       |      .
|       |      .
|       |      |
|       |      +--- P00K
|       |
|       +--- Hard
|
+-- ENV_NAME_1
.
.
|
+-- ENV_NAME_N
```

