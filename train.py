# -*- coding: utf-8 -*-
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
from utils.calculate_error import *
from datasets.datasets_list import MyDataset
from datasets.datasets_list import NYUDataset
import imageio
import imageio.core.util
from path import Path
from utils import *
from utils.logger import AverageMeter
from model.monoclip import *
import cv2
from model.losses import Criterion
from torch.optim import lr_scheduler
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Refine Depth-CLIP',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting 
parser.add_argument('--mode',type=str,default='train')
parser.add_argument('--other_method',type=str,default='MonoCLIP') # default='MonoCLIP'
parser.add_argument('--trainfile_nyu', type=str, default = "datasets/nyudepthv2_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_nyu', type=str, default = "datasets/nyudepthv2_test_files_with_gt_dense.txt")
parser.add_argument('--data_path', type=str, default = "datasets/nyu_depth_v2")
parser.add_argument('--use_dense_depth', action='store_true', help='using dense depth data for gradient loss')

# Dataloader setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=25, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', type=str, default = "NYU")

# Logging setting

# Model setting
parser.add_argument('--height', type=int, default = 480)
parser.add_argument('--width', type=int, default = 640)
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')



def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning


def main():
    args = parser.parse_args() 
    print("=> No Distributed Training")
    print('=> Index of using GPU: ', args.gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    
    model = MonoCLIP()
    print(args.height)
    train_dataset = NYUDataset()
    val_dataset = NYUDataset(train=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True)
    
    error_names = ['abs_diff', 'a1', 'a2', 'a3', 'abs_rel','log10', 'rmse']

    train_errors = AverageMeter(i=len(error_names))
    val_errors = AverageMeter(i=len(error_names))


    criterion = Criterion()
    optimizer = torch.optim.AdamW(params=model.parameters(),lr= 0.000357,weight_decay=0.1)
    scheduler = lr_scheduler.CyclicLR(optimizer=optimizer)

    epochs = args.epochs
    for epoch in epochs:
        print(f"Epochs {epoch}: ")
        model.train()
        for (rgb_img,gt_depth) in tqdm(train_dataloader):
            preds = model(rgb_img)
            optimizer.zero_grad()
            preds = list(preds)
            loss_d = 0
            for pred in preds:
                loss_d+=criterion(pred,gt_depth)
            loss_d = loss_d/len(preds)
            loss_d.backward()
            optimizer.step()
            train_err_result = compute_errors_NYU(gt=gt_depth,pred=preds)
            train_errors.update(train_err_result)
        
        print('Train metric {:.4f} ({:.4f})'.format(train_errors.val[0], train_errors.avg[0]))
        print("Train criterion: ",loss_d)
        model.eval()
        for (rgb_img,gt_depth) in tqdm(val_dataloader):
            with torch.no_grad():
                preds = model(rgb_img)
                val_err_result = compute_errors_NYU(gt=gt_depth,pred=preds)
                val_errors.update(val_err_result)
        print('Validation metric {:.4f} ({:.4f})'.format(val_errors.val[0], val_errors.avg[0]))

        scheduler.step()

if __name__ == "__main__":
    main()



