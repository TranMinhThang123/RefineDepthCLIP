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
from model.losses import SiLogLoss
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Refine Depth-CLIP',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting 
parser.add_argument('--mode',type=str,default='train')
parser.add_argument('--other_method',type=str,default='MonoCLIP') # default='MonoCLIP'
parser.add_argument('--trainfile_nyu', type=str, default = "datasets/nyudepthv2_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_nyu', type=str, default = "datasets/nyudepthv2_test_files_with_gt_dense.txt")
parser.add_argument('--data_path', type=str, default = "datasets/nyu_depth_v2")
parser.add_argument('--use_dense_depth', action='store_true', help='using dense depth data for gradient loss')
parser.add_argument("--save_dir",default="save_dir")
# Dataloader setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=25, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', type=str, default = "NYU")



# Model setting
parser.add_argument('--height', type=int, default = 480)
parser.add_argument('--width', type=int, default = 640)
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')



def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(val_loader, model, dataset = 'KITTI'):
    ##global device
    if dataset == 'KITTI':
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3','rmse','rmse_log']
    elif dataset == 'NYU':
        # error_names = ['abs_diff', 'abs_rel', 'log10', 'a1', 'a2', 'a3','rmse','rmse_log']
        error_names = ['abs_diff', 'a1', 'a2', 'a3', 'abs_rel','log10', 'rmse']
    
    elif dataset == 'Make3D':
        error_names = ['abs_diff', 'abs_rel', 'ave_log10', 'rmse']

    errors = AverageMeter(i=len(error_names))
    length = len(val_loader)
    # switch to evaluate mode
    model.eval()
    count = 0
    # max_depth=0
    for i, (rgb_data, gt_data) in enumerate(val_loader):
        if gt_data.ndim != 4 and gt_data[0] == False:
            continue
        rgb_data = rgb_data.cuda()
        gt_data = gt_data.cuda()


        input_img = rgb_data
        input_img_flip = torch.flip(input_img,[3])
        with torch.no_grad():
            
            output_depth = model(input_img)
            output_depth_flip = model(input_img_flip)
            output_depth_flip = torch.flip(output_depth_flip,[3])
            output_depth = 0.5 * (output_depth + output_depth_flip)
        if dataset == 'KITTI':
            err_result = compute_errors(gt_data, output_depth, crop=True)
        elif dataset == 'NYU':
            err_result = compute_errors_NYU(gt_data, output_depth, crop=True,idx=i)

        errors.update(err_result)
        # measure elapsed time
        # if i % 50 == 0:
        print('valid: {}/{} Abs Error {:.4f} ({:.4f})'.format(i, length, errors.val[0], errors.avg[0]))

    return errors.avg,error_names




def main():
    args = parser.parse_args() 

    
    model = MonoCLIP()
    train_dataset = NYUDataset(trainfile_nyu=args.trainfile_nyu)
    val_dataset = NYUDataset(train=False,testfile_nyu=args.testfile_nyu)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True)
    
    error_names = ['abs_diff', 'a1', 'a2', 'a3', 'abs_rel','log10', 'rmse']


    criterion = SiLogLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr= 0.003)
    epochs = args.epochs
    path = args.save_dir

    print("="*120)
    print("="*50,"START TRAINING","="*54)
    print("="*120)
    print("\n")

    losses = [torch.inf]
    best_loss = torch.inf
    for epoch in range(epochs):
        print("*"*120)
        print(f"Epochs {epoch}/{epochs}: ")
        model.train()
        for (train_rgb_img,train_gt_depths) in tqdm(train_dataloader,ncols=75,desc="Training : "):
            
            train_rgb_img = train_rgb_img.to(device)
            train_gt_depths = train_gt_depths.to(device)

            preds = model(train_rgb_img)
            
            optimizer.zero_grad()
            

            pred_list = list(preds)
            gt_depth_list = list(train_gt_depths)
            
            loss_d = 0

            for pred,gt_depth in zip(pred_list,gt_depth_list):
                loss_d+=criterion(pred,gt_depth)
            
            loss_d = loss_d/len(pred_list)

            losses.append(loss_d.item())

            loss_d.backward()
            
            optimizer.step()
            

        if loss_d<best_loss:
            print(f"\nLoss decrease from {best_loss} to {loss_d}, save best model to {path}")
            best_loss=loss_d
            torch.save(model.state_dict(),f"{path}/model2.pt")
            torch.save(optimizer.state_dict(),f"{path}/optimizer2.pt")

        print("Total train loss: {:.4f}".format(loss_d.item()))
        # scheduler.step(metrics=loss_d)
        model.eval()
        errors, error_names = validate(val_dataloader,model,"NYU")
        print("")
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[0:len(error_names)], errors[0:len(errors)]))
        print(' * Avg {}'.format(error_string))
        print("")
        
    print("\n"+"="*120)
    print("="*50,"END TRAINING","="*56)
    print("="*120)

if __name__ == "__main__":
    main()



