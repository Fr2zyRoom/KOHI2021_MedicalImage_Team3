import numpy as np


def normalize(arr):
    tmp = (arr - arr.min())/(arr.max()-arr.min())*255
    return tmp.astype(np.uint8)


def visualize_grayscale(arr):
    tmp = normalize(arr)
    return np.stack([tmp, tmp, tmp], axis=-1)
