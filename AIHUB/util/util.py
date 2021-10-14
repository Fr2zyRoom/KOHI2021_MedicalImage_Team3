import os
import numpy as np
import glob

FILE_EXTENSION = ['.png', '.PNG', '.jpg', '.JPG', '.dcm', '.DCM', '.raw', '.RAW', '.svs', '.SVS']
IMG_EXTENSION = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
DCM_EXTENSION = ['.dcm', '.DCM']
RAW_EXTENSION = ['.raw', '.RAW']
NIFTI_EXTENSION = ['.nii']
NP_EXTENSION = ['.npy']

common_dir = '/home/ncp/workspace/202002n050/050.신경계 질환 관련 임상 및 진료 데이터'


def check_extension(filename, extension_ls=FILE_EXTENSION):
    return any(filename.endswith(extension) for extension in extension_ls)


def load_file_path(folder_path, extension_ls=FILE_EXTENSION, all_sub_folders=False):
    """find 'IMG_EXTENSION' file paths in folder.
    
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

    return file_paths[:]


def gen_new_dir(new_dir):
    try: 
        if not os.path.exists(new_dir): 
            os.makedirs(new_dir) 
            #print(f"New directory!: {new_dir}")
    except OSError: 
        print("Error: Failed to create the directory.")


def find_aihub_img_label_dirs(fname, mod='train'):
    if mod == 'train':
        img_dir = os.path.join(common_dir, '01.데이터/1.Training/원천데이터', fname, 'init/image')
        mask_dir = os.path.join(common_dir, '01.데이터/1.Training/라벨링데이터', fname, 'init/mask')
    elif mod == 'val':
        img_dir = os.path.join(common_dir, '01.데이터/2.Validation/원천데이터', fname, 'init/image')
        mask_dir = os.path.join(common_dir, '01.데이터/2.Validation/라벨링데이터', fname, 'init/mask')
    else:
        return None
    return [img_dir, mask_dir]


def pair_img_mask_path(fname, mod='train'):
    img_dir, mask_dir = find_aihub_img_label_dirs(fname, mod)
    img_path_ls = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    if len(img_path_ls) == 0:
        return None
    img_path_dict = {os.path.splitext(os.path.basename(p))[0]:p for p in img_path_ls}
    if os.path.isdir(mask_dir):
        mask_path_ls = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        mask_path_dict = {os.path.splitext(os.path.basename(p))[0]:p for p in mask_path_ls}
    else:
        mask_path_dict = {}
    paired_list = []
    for imgnum, imgpath in img_path_dict.items():
        paired_list.append([imgpath, mask_path_dict.get(imgnum)])
    return paired_list


def find_aihub_img_label_paths(common_dir, mod='train'):
    if mod=='train':
        data_dir = os.path.join(common_dir, '01.데이터/1.Training/원천데이터')
    elif mod=='val':
        data_dir = os.path.join(common_dir, '01.데이터/2.Validation/원천데이터')
        
    _fname = os.listdir(data_dir)
    _fname = [p for p in _fname if os.path.isdir(os.path.join(data_dir, p))]
    paths_list = []
    for fname in _fname:
        tmp = pair_img_mask_path(fname, mod)
        if tmp:
            for p in tmp:
                paths_list.append(p)
    img_list, mask_list = list(zip(*paths_list))
    return img_list, mask_list
