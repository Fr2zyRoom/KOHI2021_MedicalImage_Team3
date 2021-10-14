import numpy as np
import PIL.Image as Image

def img_loader(img_path):
    return np.expand_dims(np.array(Image.open(img_path)), axis=-1)
def mask_loader(mask_path):
    return np.expand_dims(np.where(np.array(Image.open(mask_path)),1,0), axis=-1).astype(np.uint8)
