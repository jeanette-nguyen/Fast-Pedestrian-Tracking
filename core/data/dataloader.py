""" Load data

"""

import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from core.data.d_util import Transform, preprocess, read_image
from utils.constants import *


def get_dataloader(data_dir, batch_size=1, shuffle=True, mode=TRAIN,
                   num_workers=4, set_id='set00'):
    dataset = FastTrackDataset(mode=mode, data_dir=data_dir, set_id=set_id)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True)
    return loader

class FastTrackDataset(object):
    """Bounding box dataset for Caltech Pedestrian


    Dataset returns...

    Bounding boxes are packed into...

    Labels are packed into...
    """
    def __init__(self, data_dir, mode, set_id):
        self.mode = mode
        csv_file = os.path.join(data_dir, 'data_{}.csv'.format(self.mode))
        data = pd.read_csv(csv_file)
        data = data[data[Col.SET] == set_id]
        self.data = data[data[Col.N_LABELS] != 0].reset_index(drop=True)

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
        image, bboxes, label = self.get_example(index)

        # if self.mode != TEST:
        #     image, bboxes = Transform()
        # else:
        #     image = preprocess()

        batch = {Bat.IMG: image,
                 Bat.BBOX: bboxes,
                 Bat.LBL: label}

        return batch

    def __len__(self):
        return len(self.data)

    def get_example(self, index):
        image_filename = self.data.loc[index, Col.IMAGES]
        image = read_image(image_filename)

        bboxes = eval(self.data.loc[index, Col.COORD])
        bboxes = np.stack(bboxes).astype(np.float32)
        label = np.array(0)
        return image, bboxes, label


if __name__ == '__main__':
    batch_size = 1
    data_dir = '/Users/ktl014/PycharmProjects/ece285/Fast-Pedestrian-Tracking/data'
    loader = get_dataloader(data_dir=data_dir,
                            set_id='set00',
                            batch_size=batch_size,
                            num_workers=2,
                            mode=TRAIN)
    print(len(loader.dataset))
    for i, batch in enumerate (loader):
        img = batch[Bat.IMG].numpy ()
        bbox = batch[Bat.BBOX].numpy()
        # lbl = batch[Bat.LBL].numpy ()
        print(img.shape, img.min(), img.max(), img.dtype)
        print(bbox.shape, bbox.min(), bbox.max(), bbox.dtype)
        # print(lbl.shape, lbl.min(), lbl.max(), lbl.dtype)
        if i == 10:
            break
