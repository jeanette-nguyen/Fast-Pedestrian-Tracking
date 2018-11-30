import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

import VGG16
from utils import network as network
from utils import bbox as bbox
from utils import config as cfg
from utils import proposal_layer as proposal_layer


# def reshape(x,d):
#     x = x.view(x.size(0), d, x.size(1)*x.size(2)/d, x.size(3))
#     return x

class RPN(nn.Module):
    """
    Region proposal network (RPN) of the Faster-RCNN. It is composed of the weights from the 
    shared weights from the VGG16 network as well as its own unique weights as well.

    Attributes
    ----------
    anchor_scales:
        Scaling factor for anchor boxes
    features:
        The shared network
    conv1: 
        Convolutional layer to 
    """
    def __init__(self, pretrained=False):
        super(RPN, self).__init__()
        bn = cfg.bn
        self.bf = 10 #balancing factor
        self.img_size = cfg.img_size
        self.feat_stride = cfg.rpn_feat_stride
        self.anchor_scales = cfg.anchor_scales
        self.anchor_ratios = cfg.anchor_ratios
        self.base_anchors = bbox.generate_base_anchors()
        self.training = cfg.training


        self.conv1 = network.Conv2d(512, 512, 3, 1, relu=True, bn=bn)

        self.cls_conv = network.Conv2d(512, 
                                       len(self.anchor_scales) * len(self.anchor_ratios) * 2, 
                                       1, 
                                       0, 
                                       bn=bn)
        self.bbox_conv = network.Conv2d(512, 
                                        len(self.anchor_scales) * len(self.anchor_ratios) * 4, 
                                        1, 
                                        0, 
                                        bn=bn)

        
        self.proposal_layer = proposal_layer.ProposalLayer(self)

        # loss
        self.cls_loss = None
        self.bbox_loss = None

        if pretrained:
            self.init_pretrained_model()

    # @property ?
    # def loss(self):

    #     self._loss = self.cls_loss + self.bf*self.bbox_loss
    #     return self._loss
    
    """
    Initialize the unique RPN weights pretrained weights. 
    """
    def init_pretrained_model(self,file = 'VGGnet_fast_rcnn_iter_70000.h5'):
        if (file == 'VGGnet_fast_rcnn_iter_70000.h5'):
            params = h5py.File(file, 'r')
            self.cls_conv.bias = nn.Parameter(torch.from_numpy(params['rpn.score_conv.conv.bias'][()]))
            self.cls_conv.weight = nn.Parameter(torch.from_numpy(params['rpn.score_conv.conv.weight'][()]))
            self.bbox_conv.bias = nn.Parameter(torch.from_numpy(params['rpn.bbox_conv.conv.bias'][()]))
            self.bbox_conv.weight = nn.Parameter(torch.from_numpy(params['rpn.bbox_conv.conv.weight'][()]))
            self.conv1.bias = nn.Parameter(torch.from_numpy(params['rpn.conv1.conv.bias'][()]))
            self.conv1.weight = nn.Parameter(torch.from_numpy(params['rpn.conv1.conv.weight'][()]))
 

    def forward(self, x):
        """
        n = batch size 
        h = height of feature map after shared network
        w = width of feature map after shared network
        """
        #[n, c_out, h, w]
        n, _, h, w = x.shape

        #Get all base anchors in the original input image
        anchors = bbox.generate_shifted_anchors(self.base_anchors,h,w)
        
        x = self.conv1(x)

        # Classification 
        cls_score = self.cls_conv(x)
        cls_score = cls_score.permute(0,2,3,1).contiguous() 
        #[n, h, w, len(ratios)xlen(scales)x2]
        
        fg_score = F.softmax(cls_score.view(n,h,w,self.base_anchors.shape[0],2),dim=4)
        fg_score = fg_score[:,:,:,:,1].contiguous()
        fg_score = fg_score.view(n, -1) #(n, number of base anchors per image)

        cls_score = cls_score.view(n, -1, 2)

        # Bounding box offsets to apply to anchors to obtain region proposals
        bbox_offsets = self.bbox_conv(x)
        bbox_offsets = bbox_offsets.permute(0,2,3,1).contiguous()
        bbox_offsets = bbox_offsets.view(n, -1, 4) # -1 -> len(scales)*len(ratios)


        # Generate RoIs
        rois = []
        roi_indices = []
        for i in range(n):
            # Get all regional proposals for the input image i
            roi = self.proposal_layer(bbox_offsets[i,:,:].data.numpy(), 
                                      fg_score[i,:].data.numpy(),
                                      anchors,
                                      self.img_size)
            rois.append(roi)
            roi_indices.append(i*np.ones((len(roi),)))
        rois = np.concatenate(rois,axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)

        return rois, roi_indices




        



