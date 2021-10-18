import numpy as np


def img_loader(img_path):
    return np.expand_dims(np.load(img_path), axis=-1)
def mask_loader(mask_path):
    return np.expand_dims(np.load(mask_path), axis=-1)
