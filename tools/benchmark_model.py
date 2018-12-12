"""Benchmark Faster RCNN Model """
from __future__ import  absolute_import

# Standard dist imports
import cupy as cp
import logging
import os
import time

# Third party imports
import ipdb
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Project level imports
from core.logger import Logger
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils.eval_tool import AverageMeter

# Module level constants
FPS = 'fps'

def benchmark(benchmarker, dataloader, faster_rcnn, test_num=1000):
    logger = logging.getLogger(__name__)
    Logger.section_break(title='Benchmark Begin')

    since = time.time()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in enumerate(dataloader):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs,
                                                                       [sizes])

        benchmarker[FPS].update(time.time() - since)

        if ii % 100 == 0:
            logger.info(f'{ii}: {benchmarker[FPS]}')

        if ii == test_num: break

    return benchmarker

def main(**kwargs):
    opt._parse(kwargs)
    # Initialize Logger
    if opt.benchmark_path is None:
        timestr = time.strftime('%m%d%H%M')
        benchmark_path = f'logs/fasterrcnn_{timestr}'
        for k_, v_ in kwargs.items():
            benchmark_path += f'_{v_}'
        benchmark_path += '.log'

    Logger(benchmark_path, logging.INFO)
    logger = logging.getLogger(__name__)
    Logger.section_break(title='Benchmark Model')
    logger.info(f'User Arguments\n{opt._state_dict()}')


    # Load dataset
    dataset = Dataset(opt)
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    logger.info(f"DATASET SIZE: {len(dataloader)}")
    logger.info("Using Mask VGG") if opt.mask else logger.info("Using normal VGG16")


    # Construct model
    faster_rcnn = FasterRCNNVGG16(mask=opt.mask)
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    Logger.section_break(title='Model')
    logger.info(str(faster_rcnn))


    # Resume from a checkpoint
    if opt.load_path:
        assert os.path.isfile(opt.load_path),\
            'Checkpoint {} does not exist.'.format(opt.load_path)

        trainer.load(opt.load_path)
        Logger.section_break('Checkpoint')
        logger.info("Loaded checkpoint '{}' (epoch X)".format(opt.load_path))


    # Benchmark dataset
    benchmarker = {FPS: AverageMeter()}
    result = benchmark(benchmarker, dataloader, faster_rcnn, test_num=1000)

    #


if __name__ == '__main__':
    main()
    # import fire
    #
    # fire.Fire()
