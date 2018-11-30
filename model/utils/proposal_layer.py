import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

from utils import config as cfg
from utils import bbox as bbox

class ProposalLayer(object):
    """
    The layer of the RPN that will generate the region proposals.
    """
    def __init__(self, parent_model):
        #super(ProposalLayer, self).__init__()
        #self.anchors = generate_anchors(scales, ratios, base_size)
        #self.n_anchors = self.anchors.shape[0]
        self.training = cfg.training
        self.feat_stride = cfg.rpn_feat_stride
        self.rpn_thresh_high = cfg.rpn_thresh_high
        self.rpn_thresh_low = cfg.rpn_thresh_low
        self.pre_nms_train = cfg.rpn_pre_nms_train
        self.post_nms_train = cfg.rpn_post_nms_train
        self.pre_nms_test = cfg.rpn_pre_nms_test
        self.post_nms_test = cfg.rpn_post_nms_test
        self.min_size = cfg.rpn_min_size      

    def __call__(self, locs, scores, anchors, img_size, scale=1):
        """
        A call to the class will take in the base anchors and offset each
        one to generate the region proposal.

        Attributes
        ----------
        locs: np array
            The offsets from the base anchor the region proposal should have.
            It consists of len(scales)*len(ratios) different offsets, one
            for each different base anchors which have same center
        scores: np array
            Each index indicates the probability that an object is contained
            in an anchor
            Shape is (1, number of base anchors per image)
        anchors: np array
            All possible region proposals
        img_size: (haven't defined yet)
            The size of the input image

        Returns
        -------
        """
        if self.training:
            pre_nms = self.pre_nms_train
            post_nms = self.post_nms_train
        else:
            pre_nms = self.pre_nms_test
            post_nms = self.post_nms_test

        min_size = cfg.min_size
        rois = bbox.loc2bbox(anchors, locs) 
        #shape is (R, 4) - dim 2 is [y_min, x_min, y_max, x_max]

        # clip rois to be in image 
        rois[:, slice(0,4,2)] = np.clip(rois, 0, img_size[0]) #height
        rois[:, slice(1,4,2)] = np.clip(rois, 0, img_size[1]) #width

        #remove rois that are < minimum size in height or width
        hs = rois[:, 2]-rois[:,0]
        ws = rois[:,3]-rois[:,1]
        keep = np.where(hs[:, np.newaxis] >= min_size and 
                        ww[:, np.newaxis] >= min_size)[0]
        rois = rois[keep, :]
        scores = scores[keep]
        orders = scores.ravel().argsort()


        #rois = rois[orders[:-1*pre_nms], :] #?
        rois = rois[orders[:pre_nms], :]

        keep = non_maximum_suppression(np.ascontiguousarray(np.asarray(rois)),
                                       self.rpn_thresh_high)
        keep = keep[:post_nms]
        rois = rois[keep]
        return rois


def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    n_bbox = bbox.shape[0]

    if score is not None:
        order = score.argsort()[::-1].astype(np.int32)
    else:
        order = np.arange(n_bbox, dtype=np.int32)

    sorted_bbox = bbox[order, :]
    selec, n_selec = _call_nms_kernel(sorted_bbox, thresh)
    selec = selec[:n_selec]
    selec = order[selec]
    if limit is not None:
        selec = selec[:limit]
    return np.asnumpy(selec)




