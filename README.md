# Depth Estimation with RefineDepthCLIP
## Architecture

  This model harness the ability of cross-modal model CLIP and modify it with U-Net decoder to solve problem in monocular depth estimation 
  


  
  ![alt text](https://github.com/TranMinhThang123/RefineDepthCLIP/blob/new_fix/assets/Architecture.png)


## Prepare datasets

1. Prepare NYUDepthV2 datasets following [GLPDepth](https://github.com/vinvino02/GLPDepth) and [BTS](https://github.com/cleinc/bts/tree/master).

```
mkdir nyu_depth_v2
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./nyu_depth_v2/official_splits/
```

Download sync.zip provided by the authors of BTS from this [url](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view) and unzip in `./nyu_depth_v2` folder. 

2. Prepare KITTI datasets following [GLPDepth](https://github.com/vinvino02/GLPDepth) and [BTS](https://github.com/cleinc/bts/tree/master).


Your datasets directory should be:

```
│nyu_depth_v2/
├──official_splits/
│  ├── test
│  ├── train
├──sync/

│kitti/
├──data_depth_annotated/
├──raw_data/
├──val_selection_cropped/
```

## Results and Fine-tuned Models

| NYUv2 | RMSE | d1 | d2 | d3 | REL | Fine-tuned Model |
|-------------------|-------|-------|--------|--------|-------|-------|
| **RefineDepthCLIP** | 1.172 | 0.388 | 0.702 | 0.853 | 0.931 |[Google drive](https://drive.google.com/file/d/1w3ba7mMj6qS9-FndU5HJBP3YSKZKMUXe/view?usp=sharing) |

## Training

Run the following instuction to train the RefineDepthCLIP model.

Train the RefineDepthCLIP model with 1 RTX 3090 GPU on NYUv2 dataset:
```
python train.py 
```

## Evaluation
Command format:
```
bash eval.sh 
```
