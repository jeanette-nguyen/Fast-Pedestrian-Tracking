""" """
# Standard dist imports
import os

# Third party imports
import numpy as np
import pandas as pd

# Project level imports
from .util import read_image
from utils.constants import *

# Module level constants
CLS_IDX = {'0': 1, 'person': 2, 'people':3, 'person?':4, 'person-fa':2}

class CaltechBboxDataset:
    """Bounding box dataset for Caltech Pedestrian"""

    def __init__(self, data_dir, split=TRAIN, set_id='set00'):
        self.split = split
        csv_file = os.path.join(data_dir, 'data_{}.csv'.format(self.split))
        data = pd.read_csv(csv_file)
        data = data[data[Col.SET] == set_id]
        self.data = data[data[Col.N_LABELS] != 0].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def get_example(self, index):
        image_filename = self.data.loc[index, Col.IMAGES]
        image = read_image(image_filename)

        bboxes = np.array(eval(self.data.loc[index, Col.COORD])) #k by 4 [x_min, y_min, width, height]
        bb_copy = bboxes.copy()
        
        # Reshape to be [y_min, x_min, y_max, x_max]
        
        bb_copy[:, 2] += bb_copy[:, 0]
        bb_copy[:, 3] += bb_copy[:, 1]
        bboxes[:, 0], bboxes[:, 1] = bb_copy[:, 1], bb_copy[:, 0]
        bboxes[:, 2], bboxes[:, 3] = bb_copy[:, 3], bb_copy[:, 2]


        label = eval(self.data.loc[index, Col.LABEL])
        
        bboxes = np.stack(bboxes).astype(np.float32)
        label = np.array([CLS_IDX[i] for i in label])
        return image, bboxes, label