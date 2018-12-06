"""Data utilities"""

from PIL import Image

import numpy as np

from utils.constants import *

def read_image(path, dtype=np.float32, color=True):
    """ Readn image from a file

    This function reads an image from the given file. The image returned is
    CHW format and the range of its value is [0, 255]. If color = True,
    the order of the channels is RGB

    Args:
        path: (str) A path to the image file
        dtype: The type of array. The default is numpy.float32
        color: (bool) This flag determines teh number of channels.
            If true, the number of channels is three, which is the default
            behavior.
            If false, the function returns a grayscale image.

    Returns:
        numpy.ndarray: An image

    """
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def preprocess():
    pass

class Transform():
    pass