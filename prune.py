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
from model.compression import prune_utils
from train import eval

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

def train(opt, faster_rcnn, dataloader, test_dataloader, trainer, lr_, best_map):
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for ii, (img, bbox_, label_, scale) in pbar:
            scale = at.scalar(scale)
            
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            losses = trainer.train_step(img, bbox, label, scale, prune_train=True)
            if ii % 100 == 0:
                rpnloc = losses[0].cpu().data.numpy()
                rpncls = losses[1].cpu().data.numpy()
                roiloc = losses[2].cpu().data.numpy()
                roicls = losses[3].cpu().data.numpy()
                tot = losses[4].cpu().data.numpy()
                pbar.set_description(f"""Epoch: {epoch} | Batch: {ii} | RPNLoc Loss: {rpnloc:.4f}"""
                                    f""" | RPNclc Loss: {rpncls:.4f} | ROIloc Loss: {roiloc:.4f}"""
                                    f""" | ROIclc Loss: {roicls:.4f} | Total Loss: {tot:.4f}""")
            if (ii + 1) % 1000 == 0:
                print(trainer.get_meter_data())
                try:
                    ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                    gt_img = vis_bbox(ori_img_,
                                        at.tonumpy(bbox_[0]),
                                        at.tonumpy(label_[0]))
                    plt.show()

                    # plot predicti bboxes
                    _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                    pred_img = vis_bbox(ori_img_,
                                        at.tonumpy(_bboxes[0]),
                                        at.tonumpy(_labels[0]).reshape(-1),
                                        at.tonumpy(_scores[0]))
                    plt.show()
                except:
                    print("Cannot display images")
                
                
        eval_result = eval(test_dataloader, faster_rcnn, test_num=1000)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_), str(eval_result['map']),
                                                        str(trainer.get_meter_data()))
        print("Evaluation Results: ")
        print(log_info)
        print("\n\n")
        
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break


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