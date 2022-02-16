import os
import numpy as np
import pandas as pd
import glob
from util.util import *

def find_dwi_adc_dir(img_folder_dir, fname):
    dwi_folder_dir = os.path.join(img_folder_dir, fname, 'dwi')
    adc_folder_dir = os.path.join(img_folder_dir, fname, 'adc')
    if (os.path.isdir(dwi_folder_dir)) & (os.path.isdir(adc_folder_dir)):
        return dwi_folder_dir, adc_folder_dir
    else:
        return None


def find_mask_dir(mask_folder_dir, fname):
    mask_folder_dir = os.path.join(mask_folder_dir, fname, 'mask')
    if (os.path.isdir(mask_folder_dir)):
        return mask_folder_dir
    else:
        return None


def pair_dwi_adc_img_mask_path(img_folder_dir, mask_folder_dir):
    img_mask_path_dict = {}
    for fname in sorted(os.listdir(img_folder_dir)):
        dwi_adc_dir = find_dwi_adc_dir(img_folder_dir, fname)
        mask_dir = find_mask_dir(mask_folder_dir, fname)
        if dwi_adc_dir:
            if mask_dir:
                dwi_folder_dir, adc_folder_dir = dwi_adc_dir
                dwi_path_ls = sorted(load_file_path(dwi_folder_dir, IMG_EXTENSION))
                adc_path_ls = sorted(load_file_path(adc_folder_dir, IMG_EXTENSION))
                img_path_ls = list(zip(dwi_path_ls,adc_path_ls))
                mask_path_ls = sorted(load_file_path(mask_dir, IMG_EXTENSION))
                img_mask_path_dict[fname] = [img_path_ls, mask_path_ls]
    return img_mask_path_dict


def pair_dwi_adc_img_path(img_folder_dir):
    img_path_dict = {}
    for fname in sorted(os.listdir(img_folder_dir)):
        dwi_adc_dir = find_dwi_adc_dir(img_folder_dir, fname)
        if dwi_adc_dir:
            dwi_folder_dir, adc_folder_dir = dwi_adc_dir
            dwi_path_ls = sorted(load_file_path(dwi_folder_dir, IMG_EXTENSION))
            adc_path_ls = sorted(load_file_path(adc_folder_dir, IMG_EXTENSION))
            img_path_ls = list(zip(dwi_path_ls,adc_path_ls))
            img_path_dict[fname] = img_path_ls
    return img_path_dict


def select_train_val_test(img_mask_path_dict, fname_list):
    tmp_dict = {}
    for fname in fname_list:
        if img_mask_path_dict.get(fname):
            tmp_dict[fname] = img_mask_path_dict.get(fname)
            
    return tmp_dict


def find_img_mask_paths(img_folder_dir, mask_folder_dir):
    img_mask_path_dict = pair_dwi_adc_img_mask_path(img_folder_dir, mask_folder_dir)
    
    #img_mask_path_dict_sel = select_train_val_test(img_mask_path_dict, fname_list)
    
    img_path_arr = np.concatenate([[*img_path_ls] for img_path_ls, _ in img_mask_path_dict.values()])
    mask_path_arr = np.concatenate([mask_path_ls for _, mask_path_ls in img_mask_path_dict.values()])
    return img_path_arr, mask_path_arr


def find_img_paths(img_folder_dir):
    img_path_dict = pair_dwi_adc_img_path(img_folder_dir)
    img_path_arr = np.concatenate([[img_path] for img_path_ls in img_path_dict.values() for img_path in img_path_ls])
    return img_path_arr
