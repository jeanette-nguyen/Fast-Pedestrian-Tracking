import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

import cfg
from utils import bbox as bbox

class ProposalLayer(object):
    """
    The layer of the RPN that will generate the region proposals.
    """
    def __init__(self, parent_model):
        #super(ProposalLayer, self).__init__()
        #self.anchors = generate_anchors(scales, ratios, base_size)
        #self.n_anchors = self.anchors.shape[0]
        self.feat_stride = cfg.rpn_feat_stride
        self.thresh_high = cfg.rpn_thresh_high
        self.pre_nms_train = cfg.rpn_pre_nms_train
        self.post_nms_train = cfg.rpn_post_nms_train
        self.pre_nms_test = cfg.rpn_pre_nms_test
        self.post_nms_test = cfg.rpn_post_nms_test
        self.min_size = cfg.rpn_min_size      

    def __call__(self, loc, scores, anchors, img_size):
        """
        A call to the class will take in the base anchors and offset each
        one to generate the region proposal.

        Attributes
        ----------
        loc: np array
            The offsets from the base anchor the region proposal should have.
            It consists of len(scales)*len(ratios) different offsets, one
            for each different base anchors which have same center
        scores: np array
            Each index indicates the probability that an object is contained
            in an anchor
        anchors: np array
            All possible region proposals
        img_size: (haven't defined yet)
            The size of the input image

        Returns
        -------
        """
        anchors = bbox.loc2bbox(anchor)




