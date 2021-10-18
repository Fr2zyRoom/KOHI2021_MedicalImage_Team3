import os
import numpy as np

FILE_EXTENSION = ['.png', '.PNG', '.jpg', '.JPG', '.dcm', '.DCM', '.raw', '.RAW', '.svs', '.SVS']
IMG_EXTENSION = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
DCM_EXTENSION = ['.dcm', '.DCM']
RAW_EXTENSION = ['.raw', '.RAW']
NIFTI_EXTENSION = ['.nii']
NP_EXTENSION = ['.npy']


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


def get_data_label_in(data_dir, folder_ls):
    data_folder_path_ls = [os.path.join(data_dir, fname, 'adc') for fname in folder_ls]
    label_folder_path_ls = [os.path.join(data_dir, fname, 'mask') for fname in folder_ls]
    data_path_ls = np.hstack([load_file_path(f_path, NP_EXTENSION, all_sub_folders=True) for f_path in data_folder_path_ls])
    label_path_ls = np.hstack([load_file_path(f_path, NP_EXTENSION, all_sub_folders=True) for f_path in label_folder_path_ls])
    return {"data":sorted(data_path_ls),"label":sorted(label_path_ls)}
