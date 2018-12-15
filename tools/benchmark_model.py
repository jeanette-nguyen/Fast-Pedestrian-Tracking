"""Benchmark Faster RCNN Model """
from __future__ import  absolute_import

# Standard dist imports
import cupy as cp
import argparse
import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, os.path.abspath(os.path.pardir))
import time

# Third party imports
import numpy as np
from tqdm import tqdm

# Project level imports
from core.logger import Logger
from utils.config import opt
from data.dataset import TestDataset
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils.eval_tool import AverageMeter

# Module level constants
FPS = 'fps'

parser = argparse.ArgumentParser(description="Arguments for quantization")
parser.add_argument("--load_path", type=str, metavar='DIR',
                    help="Model to benchmark")
args = parser.parse_args()


def benchmark(benchmarker, dataloader, faster_rcnn, test_num=1000):
    logger = logging.getLogger(__name__)
    Logger.section_break(title='Benchmark Begin')

    since = time.time()
    for ii, \
        (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        since = time.time()
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs,
                                                                       [sizes])

        benchmarker[FPS].update(1/(time.time() - since))

        if ii % 10 == 0:
            logger.info('{:5}: FPS {t.val:.3f} ({t.avg:.3f})'.
                        format(ii, t=benchmarker[FPS]))

        if ii == test_num:
            break

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
    dataset = TestDataset(opt, split='val')

    dataloader = data_.DataLoader(dataset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
                                       pin_memory=True
                                       )

    logger.info(f"DATASET SIZE: {len(dataloader)}")
    logger.info("Using Mask VGG") if opt.mask else logger.info("Using normal VGG16")


    # Construct model
    faster_rcnn = FasterRCNNVGG16(mask=opt.mask)
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    Logger.section_break(title='Model')
    logger.info(str(faster_rcnn))


    # Resume from a checkpoint
    if args.load_path:
        assert os.path.isfile(args.load_path),\
            'Checkpoint {} does not exist.'.format(args.load_path)

        trainer.load(opt.load_path)
        Logger.section_break('Checkpoint')
        logger.info("Loaded checkpoint '{}' (epoch X)".format(args.load_path))


    # Benchmark dataset
    fps = AverageMeter()
    benchmarker = {FPS: fps}
    result = benchmark(benchmarker, dataloader, faster_rcnn, test_num=10000)
    Logger.section_break('Benchmark completed')
    model_parameters = filter(lambda p: p.requires_grad, faster_rcnn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('[PARAMETERS] {params}'.format(params=params))
    logger.info('[RUN TIME] {time.avg:.3f} sec/frame'.format(time=result[FPS]))


if __name__ == '__main__':
    main()

