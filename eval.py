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

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
import argparse

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def eval(dataloader, faster_rcnn, test_num=10000):
    print("\nEVAL")
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, use_07_metric=True)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path")
    args = parser.parse_args()
    
    valset = TestDataset(opt,split='val')
    val_dataloader = data_.DataLoader(valset,
                                batch_size=1,
                                num_workers=opt.test_num_workers,
                                shuffle=False,
                                pin_memory=True
                                )
    
    testset = TestDataset(opt, split='test')
    test_dataloader = data_.DataLoader(testset,
                                    batch_size=1,
                                    num_workers=opt.test_num_workers,
                                    shuffle=False,
                                    pin_memory=True
                                    )
    
    print(f"VAL SET: {len(val_dataloader)} | TEST SET: {len(test_dataloader)}")
    print("Using Mask VGG") if opt.mask else print("Using normal VGG16")
    faster_rcnn = FasterRCNNVGG16(mask=opt.mask)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    best_map = 0
    lr_ = opt.lr
    
    if args.path:
        assert os.path.isfile(args.path), 'Checkpoint {} does not exist.'.format(args.path)
        checkpoint = torch.load(args.path)['other_info']
        best_map = checkpoint['best_map']
        trainer.load(args.path)

        print("="*30+"   Checkpoint   "+"="*30)
        print("Loaded checkpoint '{}' ".format(args.path, best_map))

        eval_result = eval(val_dataloader, faster_rcnn, test_num=1000)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{},lamr:{}'.format(str(lr_),
                                                  str(trainer.get_meter_data()),
                                                  str(eval_result['map']),
                                                  str(eval_result['lamr']))
        print("Evaluation Results on Validation Set: ")
        print(log_info)
        print("\n")

        test_eval_result = eval(test_dataloader, faster_rcnn, test_num=1000)
        test_lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        test_log_info = 'lr:{}, map:{}, lamr:{}'.format(str(test_lr_),
                                                                str(test_eval_result['map']),
                                                                str(test_eval_result['lamr']))
        print("Evaluation Results on Test Set of size 1000: ")
        print(test_log_info)
        print("\n\n")
    else:
        print("No checkpoint to evaluate is specified")
        
if __name__ == "__main__":
    print(opt)
    main()
