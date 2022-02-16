import torch
import os

import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

from util.util import *
from .load_data import *


def dwi_adc_loader(dwi_adc_path):
    dwi_path, adc_path = dwi_adc_path
    dwi_img = np.array(Image.open(dwi_path))
    adc_img = np.array(Image.open(adc_path))
    return np.stack([dwi_img,adc_img], axis=-1)


def mask_loader(mask_path):
    return np.expand_dims(np.array(Image.open(mask_path))>0, axis=-1).astype(np.uint8)


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
        transform_list.append(A.Normalize(mean=(0.5,0.5),  std=(0.5,0.5)))
        #transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406),  std=(0.229, 0.224, 0.225)))
        transform_list.append(ToTensorV2(transpose_mask=True))
    return A.Compose(transform_list)


class DWI_ADC_IschemicStrokeLesionSegDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 img_folder_dir, 
                 img_loader=dwi_adc_loader, 
                 augmentation=None, 
                 preprocessing=None,
    ):
        self.img_loader = img_loader
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        self.fname_list = sorted([f for f in os.listdir(img_folder_dir) if f.startswith('.') is False])
        self.img_path_arr = find_img_paths(img_folder_dir)
        
    def __getitem__(self, index):
        image = self.img_loader(self.img_path_arr[index])
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        
        return image
    
    def __len__(self):
        return len(self.img_path_arr)
