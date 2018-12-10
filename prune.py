from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from torch.utils import data as data_
import torch
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import vis_bbox
from utils.eval_tool import eval_detection_voc
import resource
from train import train
from model.compression import prune_utils

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def main():
    dataset = Dataset(opt)
    dataloader = data_.DataLoader(dataset, \
                                batch_size=1, \
                                shuffle=True, \
                                # pin_memory=True,
                                num_workers=opt.num_workers)
    testset = TestDataset(opt, split='test')
    test_dataloader = data_.DataLoader(testset,
                                    batch_size=1,
                                    num_workers=opt.test_num_workers,
                                    shuffle=False, \
                                    pin_memory=True
                                    )

    print(f"TRAIN SET: {len(dataloader)} | TEST SET: {len(test_dataloader)}")
    print("Using Mask VGG") if opt.mask else print("Using normal VGG16")
    faster_rcnn = FasterRCNNVGG16(mask=opt.mask)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    best_map = 0
    lr_ = opt.lr

    if opt.load_path:
        assert os.path.isfile(opt.load_path), 'Checkpoint {} does not exist.'.format(opt.load_path)
        checkpoint = torch.load(opt.load_path)['other_info']
        trainer.load(opt.load_path)
        print("="*30+"   Checkpoint   "+"="*30)
        print("Loaded checkpoint '{}' (epoch {})".format(opt.load_path, 1)) #no saved epoch, put in 1 for now
        if opt.prune_by_std:
            trainer.faster_rcnn.prune_by_std(opt.std_sensitivity)
        else:
            trainer.faster_rcnn.prune_by_percentile(q=opt.percentile_sensitivity)
        prune_utils.print_nonzeros(trainer.faster_rcnn)
        train(opt, faster_rcnn, dataloader, test_dataloader, trainer, lr_, best_map, prune=True)
        trainer.faster_rcnn.set_pruned()
    else:
        print("Must specify load path to pretrained model")
    

if __name__ == "__main__":
    print(opt)
    main()