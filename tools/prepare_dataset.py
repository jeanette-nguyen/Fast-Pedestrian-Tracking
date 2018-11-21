"""Prepare CalTech Pedestrian Train/Test Dataset Partitions

# Example
Run command as follows to prepare dataset:


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Standard dist imports
import argparse
import glob
import json
import logging
import os
import sys

# Third party imports
import pandas as pd

# Project level imports
from core.logger import Logger
from utils.constants import *

# Module level constants
DEBUG = True


def parse_cmds():
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--path', '-p', type=str, metavar='DIR',
                        help='Path to download dataset')
    parser.add_argument('--data-dir', dest='data_dir',
                        type=str, metavar='DIR', help='Data directory')
    args = parser.parse_args(sys.argv[1:])
    return args


def main():
    args = parse_cmds()
    if DEBUG:
        dest_dir = os.path.join(os.path.dirname(__file__), 'data')
        src_dir = '/Users/ktl014/PycharmProjects/ece285/caltech-pedestrian-dataset-converter'
    else:
        dest_dir = args.path
        src_dir = args.data_dir

    # Console logger
    log_filename = os.path.join(dest_dir, 'dataset.log')
    Logger(log_filename, logging.INFO)
    logger = logging.getLogger(__name__)
    Logger.section_break(title='Generate Dataset')

    # Initialize DatasetGenerator
    datagen = DatasetGenerator(src_dir, logger)
    datagen.generate()


class DatasetGenerator(object):
    def __init__(self, src_dir, logger):
        self.src_dir = src_dir
        self.logger = logger

    def generate(self):
        self.images = self._get_images_paths()

        self.labels = self._get_labels()


    def _get_images_paths(self):
        images = glob.glob(os.path.join(self.src_dir, 'data', 'images', '*'))
        return images

    def _get_labels(self):
        check = 'CHECKING IF GITHUB UPDATE WORKS'
        pass

if __name__ == '__main__':
    main()
