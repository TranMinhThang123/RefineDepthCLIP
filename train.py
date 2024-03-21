# -*- coding: utf-8 -*-
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
from calculate_error import *
from datasets.datasets_list import MyDataset
from datasets.datasets_list import NYUDataset
import imageio
import imageio.core.util
from path import Path
from utils import *
from logger import AverageMeter
from monoclip import *
import cv2

parser = argparse.ArgumentParser(description='Transformer-based Monocular Depth Estimation with Attention Supervision',
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
parser.add_argument('--epochs', default=0, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=24, type=int, metavar='N', help='mini-batch size')
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


def train(train_loader,val_loader, epochs,  model, dataset = 'NYU'):
    
    for epoch in range(epochs):
        for i, (rgb_data, gt_data) in enumerate(train_loader):

            input_img = rgb_data
            copy_input = np.squeeze(input_img.permute(2,3,1,0).numpy().copy())
            cv2.imwrite("vis_res/input.jpeg",copy_input)
            input_img_flip = torch.flip(input_img,[3])
            with torch.no_grad():
                
                output_depth = model(input_img)
                output_depth_flip = model(input_img_flip)
                output_depth_flip = torch.flip(output_depth_flip,[3])
                output_depth = 0.5 * (output_depth + output_depth_flip)

                output_depth = nn.functional.interpolate(output_depth, size=[480, 640], mode='bilinear', align_corners=True)
                
            if dataset == 'KITTI':
                err_result = compute_errors(gt_data, output_depth, crop=True, cap=args.cap)
            elif dataset == 'NYU':
                err_result = compute_errors_NYU(gt_data, output_depth, crop=True,idx=i)




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

    if args.mode == "train":
        train(model=model,epochs=args.epochs,val_loader=val_dataloader,train_loader=train_dataloader)
    



if __name__ == "__main__":
    main()



