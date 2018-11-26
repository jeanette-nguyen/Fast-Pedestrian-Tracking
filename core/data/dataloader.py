""" Load data

"""

import os

import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from utils.constants import *

def main()
    pass

def get_dataloader(batch_size=1, shuffle=True, mode=TRAIN, num_workers=4):
    dataset = FastTrackDataset(mode=mode)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True)
    return loader


class FastTrackDataset(object):
    def __init__(self, data_dir, mode):
        self.mode = mode
        csv_file = os.path.join(data_dir, 'data_{}.csv'.format(self.mode))
        self.data = pd.read_csv(csv_file)

    def __getitem__(self, index):
        # Read image (N --> batch size)
        # Grab coordinates (N, R, 4) R--># of bboxes per image
        # Grab labels
        # Grab original image size
        # Preprocess
        # Transform

        # Need to review train -> trainer etc.
        # Initially need to get coordinates into right shape and grab images
        # set up if condition to transform or preprocess
        pass

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    import time
    import sys

    batch_size = 64
    loader = get_dataloader(batch_size=batch_size, num_workers=2, mode='train')
    print(len(loader.dataset))
    for i, batch in enumerate (loader):
        img = batch['rgb'].numpy ()
        lbl = batch['label'].numpy ()
        print(img.shape, img.min (), img.max (), img.dtype)
        print(lbl.shape, lbl.min (), lbl.max (), lbl.dtype)
