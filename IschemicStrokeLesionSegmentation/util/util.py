FILE_EXTENSION = ['.png', '.PNG', '.jpg', '.JPG', '.dcm', '.DCM', '.raw', '.RAW', '.img', '.IMG']
NIFTI_EXTENSION = ['.nii', 'nii.gz']
DCM_EXTENSION = ['.dcm', '.DCM']
IMG_EXTENSION = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']

import os
import pandas as pd
import numpy as np
import PIL.Image as Image


def check_extension(filename, extension_ls=FILE_EXTENSION):
    return any(filename.endswith(extension) for extension in extension_ls)

def load_file_path(folder_path, extension_ls=FILE_EXTENSION, all_sub_folders=False):
    """find 'FILE_EXTENSION' file paths in folder.
    
    Parameters:
        folder_path (str) -- folder directory
        extension_ls (list) -- list of extensions
        
    Return:
        file_paths (list) -- list of 'extension_ls' file paths
    
    """
    
    file_paths = []
    assert os.path.isdir(folder_path), f'{folder_path} is not a valid directory'

    for root, _, fnames in sorted(os.walk(folder_path)):
        for fname in fnames:
            if check_extension(fname, extension_ls):
                path = os.path.join(root, fname)
                file_paths.append(path)
        if not all_sub_folders:
            break
    
    return sorted(file_paths)[:]

def gen_new_dir(new_dir):
    try: 
        if not os.path.exists(new_dir): 
            os.makedirs(new_dir) 
            #print(f"New directory!: {new_dir}")
    except OSError: 
        print("Error: Failed to create the directory.")


def normalize(img_arr):
    norm_arr = img_arr-img_arr.min()
    if norm_arr.max() != 0:
        norm_arr = norm_arr / norm_arr.max()
    norm_arr = (norm_arr*255)
    return norm_arr.astype(np.uint8)


def save_arr_to_np(arr, savepoint, fname):
    np.save(os.path.join(savepoint, fname+'.npy'), arr)


def resize_and_save_3d(im_3d, save_point):
    file_name = 0
    for im_2d in im_3d:
        resized_img = Image.fromarray(im_2d).resize((256, 256))
        resized_img.save(os.path.join(save_point, str(file_name).zfill(3)+'.png'))
        file_name += 1


def save_arr_to_png(im_2d, save_point, filename):
    Image.fromarray(im_2d).save(os.path.join(save_point, filename+'.png'))
