from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def transform_bbox(bbox_):
    """
    Transforms caltech pedestrian bbox to same format as VOC
    Args:
        bbox_: tensor of {x_min, y_min, width, height}
    Returns:
        bbox_: tensor of {y_min, x_min, y_max, x_max}
    """
    bbox_[:, :, 2] += bbox_[:, :, 0]
    bbox_[:, :, 3] += bbox_[:, :, 1]
    bbox_[:, :, 0], bbox_[:, :, 1] = bbox_[:, :, 1], bbox_[:, :, 0]
    bbox_[:, :, 2], bbox_[:, :, 3] = bbox_[:, :, 3], bbox_[:, :, 2]
    return bbox_

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

def train(opt, faster_rcnn, dataloader, val_dataloader, trainer, lr_, best_map):
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for ii, (img, bbox_, label_, scale) in pbar:
            # Currently configured to predict (y_min, x_min, y_max, x_max)
#             bbox_tmp = bbox_.clone()
#             bbox_ = transform_bbox(bbox_)
            scale = at.scalar(scale)
            
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            losses = trainer.train_step(img, bbox, label, scale)
            if ii % 100 == 0:
                rpnloc = losses[0].cpu().data.numpy()
                rpncls = losses[1].cpu().data.numpy()
                roiloc = losses[2].cpu().data.numpy()
                roicls = losses[3].cpu().data.numpy()
                tot = losses[4].cpu().data.numpy()
                pbar.set_description(f"Epoch: {epoch} | Batch: {ii} | RPNLoc Loss: {rpnloc:.4f} | RPNclc Loss: {rpncls:.4f} | ROIloc Loss: {roiloc:.4f} | ROIclc Loss: {roicls:.4f} | Total Loss: {tot:.4f}")
            if (ii + 1) % 100 == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                print(trainer.get_meter_data())
                try:
                    ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                    gt_img = visdom_bbox(ori_img_,
                                        at.tonumpy(bbox_[0]),
                                        at.tonumpy(label_[0]))
                    trainer.vis.img('gt_img', gt_img)
                    plt.show()

                    # plot predicti bboxes
                    _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                    pred_img = visdom_bbox(ori_img_,
                                        at.tonumpy(_bboxes[0]),
                                        at.tonumpy(_labels[0]).reshape(-1),
                                        at.tonumpy(_scores[0]))
                    plt.show()
                    trainer.vis.img('pred_img', pred_img)

                    # rpn confusion matrix(meter)
                    trainer.vis.text(str(trainer.rpn_cm.value().tolist()),
                                     win='rpn_cm')
                    # roi confusion matrix
                    trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf,
                                                          False).float())
                except:
                    print("Cannot display images")

        # Save after every epoch
        epoch_path = trainer.save(epoch, best_map=0,
                     #save_path='/data6/lekevin/fast_track/jeanette/Fast-Pedestrian-Tracking/checkpoints/fasterrcnn_pretrained_epoch{}'.format(epoch))
                     save_path='/data6/lekevin/fast_track/jeanette/Fast-Pedestrian-Tracking/checkpoints/deleteme_{}'.format(epoch))
                
        eval_result = eval(val_dataloader, faster_rcnn, test_num=1000)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{}, lamr:{}, loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(eval_result['lamr']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)
        print("Evaluation Results on Validation Set: ")
        print(log_info)
        print("\n\n")

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = epoch_path

        if epoch == 9:
            trainer.load(best_path, simple=False)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break

def main():
    dataset = Dataset(opt, set_id='set00', split='train')
    dataloader = data_.DataLoader(dataset, \
                                batch_size=1, \
                                shuffle=True, \
                                num_workers=opt.num_workers)

    valset = TestDataset(opt, set_id='set00', split='val')
    val_dataloader = data_.DataLoader(valset,
                                    batch_size=1,
                                    num_workers=opt.test_num_workers,
                                    shuffle=False, \
                                    pin_memory=True
                                    )

    print(f"TRAIN SET: {len(dataloader)} | VAL SET: {len(val_dataloader)}")
    print("Using Mask VGG") if opt.mask else print("Using normal VGG16")
    faster_rcnn = FasterRCNNVGG16(mask=opt.mask)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    
    if opt.load_path:
        assert os.path.isfile(opt.load_path), 'Checkpoint {} does not exist.'.format(opt.load_path)
        checkpoint = torch.load(opt.load_path)['other_info']
        if opt.use_simple:
            start_epoch = 0
            best_map = 0
        else:
            start_epoch = checkpoint['epoch']
            best_map = checkpoint['best_map']
        trainer.load(opt.load_path)
        print("="*30+"   Checkpoint   "+"="*30)
        print("Loaded checkpoint '{}' (epoch {})".format(opt.load_path, start_epoch))
    
    
    train(opt, faster_rcnn, dataloader, val_dataloader, trainer, lr_, best_map)


if __name__ == "__main__":
    print(opt)
    main()