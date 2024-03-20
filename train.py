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
parser.add_argument('--models_list_dir', type=str, default='')
parser.add_argument('--result_dir', type=str, default='')
parser.add_argument('--model_dir',type=str)
parser.add_argument('--other_method',type=str,default='MonoCLIP') # default='MonoCLIP'
parser.add_argument('--trainfile_nyu', type=str, default = "/home/rrzhang/zengzy/code/clip_depth/datasets/nyudepthv2_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_nyu', type=str, default = "datasets/nyudepthv2_test_files_with_gt_dense_2.txt")
parser.add_argument('--data_path', type=str, default = "datasets/nyu_depth_v2")
parser.add_argument('--use_dense_depth', action='store_true', help='using dense depth data for gradient loss')

# Dataloader setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--epochs', default=0, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=24, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', type=str, default = "KITTI")

# Logging setting
parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--log-metric', default='_LRDN_evaluation.csv', metavar='PATH', help='csv where to save validation metric value')
parser.add_argument('--val_in_train', action='store_true', help='validation process in training')

# Model setting
parser.add_argument('--encoder', type=str, default = "ResNext101")
parser.add_argument('--norm', type=str, default = "BN")
parser.add_argument('--act', type=str, default = "ReLU")
parser.add_argument('--height', type=int, default = 352)
parser.add_argument('--width', type=int, default = 704)
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')
parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

# Evaluation setting
parser.add_argument('--evaluate', action='store_true', help='evaluate score')
parser.add_argument('--multi_test', action='store_true', help='test all of model in the dir')
parser.add_argument('--img_save', action='store_true', help='will save test set image')
parser.add_argument('--cap', default=80.0, type=float, metavar='MaxVal', help='cap setting for kitti eval')

# GPU parallel process setting
parser.add_argument('--gpu_num', type=str, default = "0,1,2,3", help='force available gpu index')
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)

def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning


def train(args, val_loader, model, dataset = 'KITTI'):
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
    for i, (rgb_data, gt_data, dense) in enumerate(val_loader):
        if gt_data.ndim != 4 and gt_data[0] == False:
            continue
        rgb_data = rgb_data
        gt_data = gt_data


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
            copy_out_depth = np.squeeze(output_depth.permute(2,3,1,0).numpy().copy())
            background_image = plt.imread("datasets\\nyu_depth_v2\official_splits\\test\\bathroom\\rgb_00045.jpg")
            # Create a figure with the same size as the background image
            fig, ax = plt.subplots(figsize=(background_image.shape[1] / 100, background_image.shape[0] / 100))

            # Plot the background image
            ax.imshow(background_image)

            # Plot the heatmap on top of the background
            heatmap_plot = ax.imshow(copy_out_depth, cmap='viridis', interpolation='nearest', alpha=0.7)  # Adjust alpha as needed

            # Add a colorbar
            cbar = fig.colorbar(heatmap_plot)

            # Save the figure with the heatmap and background
            fig.savefig('vis_res/heatmap.png')
            cv2.imwrite("vis_res/output_depth.jpg",copy_out_depth)
        if dataset == 'KITTI':
            err_result = compute_errors(gt_data, output_depth, crop=True, cap=args.cap)
        elif dataset == 'NYU':
            err_result = compute_errors_NYU(gt_data, output_depth, crop=True,idx=i)

        errors.update(err_result)
        # measure elapsed time
        if i % 50 == 0:
            print('valid: {}/{} Abs Error {:.4f} ({:.4f})'.format(i, length, errors.val[0], errors.avg[0]))

    return errors.avg,error_names


def main():
    args = parser.parse_args() 
    print("=> No Distributed Training")
    print('=> Index of using GPU: ', args.gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    
    model = MonoCLIP()
    print(args.height)
    train_dataset = NYUDataset(args)

if __name__ == "__main__":
    main()



