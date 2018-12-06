"""Draw bounding box annotations

Creates videos from frames to view bounding boxes.
Images and annotations are retrieved from csv file
generated from prepare_dataset.py

# Example
Set module level constants
    PHASE = TRAIN # Options [TRAIN, VAL, TEST]
    SET_ID = 'set00' # Options 'setXX' XX~0-10

Run command as follows to plot annotations:
$   python plot_annotations.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Standard dist imports
import os

# Third party imports
import cv2
import pandas as pd

# Project level imports
from utils.constants import *

# Module level constants
DEBUG = False
PHASE = TRAIN
SET_ID = 'set00'

def main():
    src_dir = os.path.abspath(os.pardir)
    dest_dir = os.path.join(os.path.abspath(os.pardir), 'data/plots')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Read in dataset
    filename = os.path.join(src_dir, 'data/data_{}.csv'.format(PHASE))
    df = pd.read_csv(filename)
    df = df[df[Col.VALID] == True]

    # Filter for desired set_id
    df = df[df[Col.SET] == SET_ID]

    # For each video, write a video
    n_objects = 0
    grouped_videos = df.groupby(Col.VIDEO)
    for v, v_df in grouped_videos:
        # Initialize video writer
        filename = os.path.join(dest_dir, '{}_{}.avi'.format(SET_ID, v))
        v_wri = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 30,
                               (640, 480))

        v_df = v_df.sort_values(Col.FRAME)
        for idx, row in v_df.iterrows():
            # Grab coordinates and images
            img = cv2.imread(row[Col.IMAGES])
            data = eval(row[Col.COORD])

            for datum in data:
                # Draw bounding boxes
                x, y, w, h = [int(v) for v in datum]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
                n_objects += 1
            v_wri.write(img)
        v_wri.release()
        print(SET_ID, v)

if __name__ == '__main__':
    main()