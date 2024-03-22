import torch.utils.data as data
from PIL import Image
import numpy as np
from imageio import imread
import random
import torch
import time
import cv2
from PIL import ImageFile
from .transform_list import RandomCropNumpy,EnhancedCompose,RandomColor,RandomHorizontalFlip,ArrayToTensorNumpy,Normalize,CropNumpy
from torchvision import transforms
import pdb
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

def _is_pil_image(img):
    return isinstance(img, Image.Image)

class MyDataset(data.Dataset):
    def __init__(self, args, train=True, return_filename = False):
        self.use_dense_depth = args.use_dense_depth
        if train is True:
            if args.dataset == 'KITTI':
                self.datafile = args.trainfile_kitti
                self.angle_range = (-1, 1)
                self.depth_scale = 256.0
            elif args.dataset == 'NYU':
                self.datafile = args.trainfile_nyu 
                self.angle_range = (-2.5, 2.5)
                self.depth_scale = 1000.0
                args.height = 416
                args.width = 544
        else:
            if args.dataset == 'KITTI':
                self.datafile = args.testfile_kitti
                self.depth_scale = 256.0
            elif args.dataset == 'NYU':
                self.datafile = args.testfile_nyu
                self.depth_scale = 1000.0
                args.height = 416
                args.width = 544
        self.train = train
        self.transform = Transformer(args)
        self.args = args
        if train:
            self.data_path = self.args.data_path + '/train'
        else:
            if args.dataset == 'KITTI':
                self.data_path = '/model/Narci/SARPN/datasets/KITTI'
            if args.dataset == 'NYU':
                self.data_path = "datasets\\nyu_depth_v2\official_splits\\test".replace("\\",os.sep)
        self.return_filename = return_filename
        with open(self.datafile, 'r') as f:
            self.fileset = f.readlines()
        self.fileset = sorted(self.fileset)

    def __getitem__(self, index):
        divided_file = self.fileset[index].split()
        if self.args.dataset == 'KITTI':
            date = divided_file[0].split('/')[0] + '/'
        # Opening image files.   rgb: input color image, gt: sparse depth map
        rgb_file = self.data_path + '/' + divided_file[0].replace("/",os.sep)
        rgb = Image.open(rgb_file)
        gt = False
        gt_dense = False
        if (self.train is False):
            divided_file_ = divided_file[0].split('/')
            if self.args.dataset == 'KITTI':
                filename = divided_file_[1] + '_' + divided_file_[4]
            else:
                filename = divided_file_[0] + '_' + divided_file_[1]
            
            if self.args.dataset == 'KITTI':
                # Considering missing gt in Eigen split
                if divided_file[1] != 'None':
                    gt_file = self.data_path + '/data_depth_annotated/' + divided_file[1]
                    gt = Image.open(gt_file)
                    if self.use_dense_depth is True:
                        gt_dense_file = self.data_path + '/data_depth_annotated/' + divided_file[2]
                        gt_dense = Image.open(gt_dense_file)
                else:
                    pass
            elif self.args.dataset == 'NYU':
                gt_file = self.data_path + '/' + divided_file[1]
                gt = Image.open(gt_file)
                if self.use_dense_depth is True:
                    gt_dense_file = self.data_path + '/' + divided_file[2]
                    gt_dense = Image.open(gt_dense_file)

        else:
            angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
            if self.args.dataset == 'KITTI':
                gt_file = self.data_path + '/data_depth_annotated/' + divided_file[1]
                if self.use_dense_depth is True:
                    gt_dense_file = self.data_path + '/data_depth_annotated/' + divided_file[2]
            elif self.args.dataset == 'NYU':
                gt_file = self.data_path + '/' + divided_file[1]
                if self.use_dense_depth is True:
                    gt_dense_file = self.data_path + '/' + divided_file[2]
            
            gt = Image.open(gt_file)
            rgb = rgb.rotate(angle, resample=Image.BILINEAR)
            gt = gt.rotate(angle, resample=Image.NEAREST)
            if self.use_dense_depth is True:
                gt_dense = Image.open(gt_dense_file) 
                gt_dense = gt_dense.rotate(angle, resample=Image.NEAREST)

        if self.args.dataset == 'KITTI':
            h = rgb.height
            w = rgb.width
            bound_left = (w - 1216)//2
            bound_right = bound_left + 1216
            bound_top = h - 352
            bound_bottom = bound_top + 352
        elif self.args.dataset == 'NYU':
            if self.train is True:
                bound_left = 43
                bound_right = 608
                bound_top = 45
                bound_bottom = 472
            else:
                bound_left = 0
                bound_right = 640
                bound_top = 0
                bound_bottom = 480
        # print("max gt: ",np.max(np.asanyarray(gt)))
        # save for vis
        rgb = rgb.crop((bound_left,bound_top,bound_right,bound_bottom))
        rgb = np.asarray(rgb, dtype=np.float32)/255.0

        

        

        if _is_pil_image(gt):
            gt = gt.crop((bound_left,bound_top,bound_right,bound_bottom))
            gt = (np.asarray(gt, dtype=np.float32))/self.depth_scale
            gt = np.expand_dims(gt, axis=2)
            gt = np.clip(gt, 0, self.args.max_depth)
        
        rgb, gt, gt_dense = self.transform([rgb] + [gt] + [gt_dense], self.train)
        if self.return_filename is True:
            return rgb, gt, gt_dense, filename
        else:
            return rgb, gt, gt_dense

    def __len__(self):
        return len(self.fileset)


class NYUDataset(data.Dataset):
    def __init__(self,data_path="datasets/nyu_depth_v2",
                 trainfile_nyu="datasets/nyudepthv2_train_files_with_gt_dense.txt",
                 testfile_nyu="datasets/nyudepthv2_test_files_with_gt_dense.txt",
                 train=True,
                 maxdepth=80.0,
                 depthscale=1000.0) -> None:
        super().__init__()
        self.max_depth = maxdepth
        self.depth_scale = depthscale
        self.train = train
        self.data_path = data_path
        if self.train:
            self.datafile = trainfile_nyu
        else:
            self.datafile = testfile_nyu

        with open(self.datafile,'r') as f:
            self.img_label_pair = f.readlines()

        self.transformer = EnhancedCompose([
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None]
            ])

    def __getitem__(self, index):
        image_name,depth_name = self.img_label_pair[index].split()
        dataset_type = "train" if self.train else "test"
        image_path = self.data_path+f"/official_splits/{dataset_type}/"+image_name
        depth_path = self.data_path+f"/official_splits/{dataset_type}/"+depth_name

        rgb = Image.open(image_path)
        rgb = np.asarray(rgb, dtype=np.float32)/255.0
        gt = Image.open(depth_path)

        if _is_pil_image(gt):
            gt = (np.asarray(gt, dtype=np.float32))/self.depth_scale
            gt = np.expand_dims(gt, axis=2)
            gt = np.clip(gt, 0, self.max_depth)
        rgb, gt = self.transformer([rgb] + [gt] , self.train)

        return rgb,gt


class Transformer(object):
    def __init__(self, args):
        if args.dataset == 'KITTI':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height,args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                CropNumpy((args.height,args.width)),
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
        elif args.dataset == 'NYU':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height,args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.8, 1.2),brightness_mult_range=(0.75, 1.25)), None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                # CropNumpy((args.height,args.width)),
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)
