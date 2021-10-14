import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import os
from util.util import *
from data.dataloader import *

import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation(params=None):
    transform_list = []
    
    #transform_list.append(A.HorizontalFlip(p=.5))
    #transform_list.append(A.VerticalFlip(p=.5))
    #transform_list.append(A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=5, shift_limit=0.2, border_mode=0, p=.5))
    #transform_list.append(A.ShiftScaleRotate(scale_limit=0.01, rotate_limit=5, shift_limit=0., border_mode=0, p=.5))
    
    return A.Compose(transform_list)


def get_preprocessing(params=None,resize=(256,256),convert=True):
    transform_list = []
    transform_list.append(A.Resize(*resize))
    if convert:
        transform_list.append(A.Normalize(mean=(0.5,),  std=(0.5,)))
        #transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406),  std=(0.229, 0.224, 0.225)))
        transform_list.append(ToTensorV2(transpose_mask=True))
    return A.Compose(transform_list)


class AIHUB_LesionSegDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset_dir, 
                 img_loader=img_loader, 
                 mask_loader=mask_loader,
                 augmentation=None, 
                 preprocessing=None,
                 mode='train'
    ):
        self.dataset_dir = dataset_dir
        self.img_loader = img_loader
        self.mask_loader = mask_loader
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode
        self.img_path_ls, self.mask_path_ls = find_aihub_img_label_paths(common_dir, mod=self.mode)
        if self.mode != 'train':
            self.augmentation = None
        
    def __getitem__(self, index):
        image = self.img_loader(self.img_path_ls[index])
        if self.mask_path_ls[index]:  
            mask = self.mask_loader(self.mask_path_ls[index])
        else:
            mask = np.zeros_like(image)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
    
    def __len__(self):
        return len(self.img_path_ls)
