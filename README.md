# Description
- This project apply CLIP in Monocular depth estimation
# Installation
- Using conda:<br>
```
conda env create -f environment.yml
```
- Using python:<br>
```
pip install -r requirements.txt
```
# Idea
After study, we find that the text prompt is not close to depth property of input image.So we create a module to advance modify the image and bring them close to the depth map output
![affect of promt to image](https://github.com/TranMinhThang123/RefineDepthCLIP/blob/develop/vis_res/heatmap.png)
