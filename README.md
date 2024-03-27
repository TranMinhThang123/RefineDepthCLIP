# Depth Estimation with RefineDepthCLIP
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
| **MetaPrompts** | 0.223 | 0.976 | 0.997 | 0.999 | 0.061 |[Google drive](https://drive.google.com/file/d/1IBZ34fCaD7vTpr4eS7Cq-gfJWDOzx54A/view?usp=sharing) |

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
