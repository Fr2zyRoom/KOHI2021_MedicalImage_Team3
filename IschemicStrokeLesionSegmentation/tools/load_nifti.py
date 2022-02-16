
import numpy as np
import nibabel as nib


def read_nifti_file(file_path):
    """Read and load volume"""
    #Read file
    scan = nib.load(file_path)
    #Get raw data
    scan = scan.get_fdata()
    return scan.transpose(2,1,0)


def read_nifti_mask(file_path):
    scan = read_nifti_file(file_path)
    scan = scan[::-1]
    mask = np.where(scan>0, 1., 0.)
    return mask.astype(np.uint8)
