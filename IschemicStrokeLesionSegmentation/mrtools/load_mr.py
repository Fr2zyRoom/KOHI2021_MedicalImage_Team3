import os
import numpy as np
import pydicom
import scipy.ndimage
from util.util import *


def load_mr_scans(dcm_path_ls, norm_mode = '2d'):
    slices = [pydicom.read_file(dcm_path, force=True) for dcm_path in dcm_path_ls]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    
    images = np.stack([file.pixel_array for file in slices])
    if norm_mode == '2d':
        images = np.stack([normalize(image) for image in images])
    elif norm_mode == '3d':
        images = normalize(images)
    else:
        pass
    return slices, images


def resample_3d(image_3d, dsize=(36,256,256)):
    rounded_resize_factor = np.array(dsize) / image_3d.shape
    
    return scipy.ndimage.interpolation.zoom(image_3d, rounded_resize_factor, mode='nearest')
