import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from utils import network as network
from utils import config as cfg

class VGG16(nn.Module):
    """
    The shared convolutional network for the RPN network and the classifier network. 
    All the convolutional layers (model features) are used, while the fully connected layers
    (model classifier) are disregarded.

    Parameters
    ----------
    bn : boolean
        Indicates whether to load weights from the VGG16 network trained with 
        batch normalization
    pretrained : boolean
        Indicates whether to use load the weights from the Faster RCNN 
        network (True) or the weights from the VGG16 network (False)
    fine_tune : boolean
        Indicates whether to fine tune the pretrained VGG16 network

    Attributes
    ----------
    self.conv1:
        Two convolution layers, each followed by ReLU, and a max pooling
    self.conv2:
        Two convolution layers, each followed by ReLU, and a max pooling
    self.conv3:
        Three convolution layers, each followed by ReLU, and a max pooling
    self.conv4:
        Three convolution layers, each followed by ReLU, and a max pooling
    self.conv5
        Three convolution layers, each followed by ReLU
    """
    def __init__(self, relu=True, pretrained=False):
        super(VGG16, self).__init__()
        bn = cfg.bn
        # note: do note need classifier
        if not pretrained:
            if bn:
                model = models.vgg16_bn(pretrained=True)
                self.conv1 = nn.Sequential(*list(model.features.children())[:7])
                self.conv2 = nn.Sequential(*list(model.features.children())[7:14])
                self.conv3 = nn.Sequential(*list(model.features.children())[14:24])
                self.conv4 = nn.Sequential(*list(model.features.children())[24:34])
                self.conv5 = nn.Sequential(*list(model.features.children())[34:43])

            else:
                model = models.vgg16(pretrained=True)
                self.conv1 = nn.Sequential(*list(model.features.children())[:5])
                self.conv2 = nn.Sequential(*list(model.features.children())[5:10])
                self.conv3 = nn.Sequential(*list(model.features.children())[10:17])
                self.conv4 = nn.Sequential(*list(model.features.children())[17:24])
                self.conv5 = nn.Sequential(*list(model.features.children())[24:30])
                
        else:
            self.conv1 = list((network.Conv2d(3, 64, 3, 1, bn=bn, relu=relu),
                               network.Conv2d(64, 64, 3, 1, bn=bn, relu=relu),
                               nn.MaxPool2d(2)))
            self.conv2 = list((network.Conv2d(64, 128, 3, 1, bn=bn, relu=relu),
                               network.Conv2d(128, 128, 3, 1, bn=bn, relu=relu),
                               nn.MaxPool2d(2)))
            self.conv3 = list((network.Conv2d(128, 256, 3, 1, bn=bn, relu=relu),
                               network.Conv2d(256, 256, 3, 1, bn=bn, relu=relu),
                               network.Conv2d(256, 256, 3, 1, bn=bn, relu=relu),
                               nn.MaxPool2d(2)))
            self.conv4 = list((network.Conv2d(256, 512, 3, 1, bn=bn, relu=relu),
                               network.Conv2d(512, 512, 3, 1, bn=bn, relu=relu),
                               network.Conv2d(512, 512, 3, 1, bn=bn, relu=relu),
                               nn.MaxPool2d(2)))
            self.conv5 = list((network.Conv2d(512, 512, 3, 1, bn=bn, relu=relu),
                               network.Conv2d(512, 512, 3, 1, bn=bn, relu=relu),
                               network.Conv2d(512, 512, 3, 1, bn=bn, relu=relu)))
            self.init_pretrained_model()

        network.freeze(self.conv1)
        network.freeze(self.conv2)

        
    def init_pretrained_model(self, file = 'VGGnet_fast_rcnn_iter_70000.h5'):
        """
        Initialize weights for VGG16 network using the final weights from a Faster-RCNN network
        trained on Pascal VOC 2007.

        Parameters
        ----------
        file: str
            name of file where the weights are stored
        """
        
        if (file == 'VGGnet_fast_rcnn_iter_70000.h5'):
            params = h5py.File(file, 'r')

            self.conv1[0].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv1.0.conv.bias'][()]))
            self.conv1[0].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv1.0.conv.weight'][()]))
            self.conv1[1].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv1.1.conv.bias'][()]))
            self.conv1[1].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv1.1.conv.weight'][()]))

            self.conv2[0].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv2.0.conv.bias'][()]))
            self.conv2[0].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv2.0.conv.weight'][()]))
            self.conv2[1].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv2.1.conv.bias'][()]))
            self.conv2[1].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv2.1.conv.weight'][()]))

            self.conv3[0].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv3.0.conv.bias'][()]))
            self.conv3[0].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv3.0.conv.weight'][()]))
            self.conv3[1].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv3.1.conv.bias'][()]))
            self.conv3[1].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv3.1.conv.weight'][()]))
            self.conv3[2].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv3.2.conv.bias'][()]))
            self.conv3[2].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv3.2.conv.weight'][()]))

            self.conv4[0].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv4.0.conv.bias'][()]))
            self.conv4[0].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv4.0.conv.weight'][()]))
            self.conv4[1].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv4.1.conv.bias'][()]))
            self.conv4[1].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv4.1.conv.weight'][()]))
            self.conv4[2].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv4.2.conv.bias'][()]))
            self.conv4[2].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv4.2.conv.weight'][()]))

            self.conv5[0].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv5.0.conv.bias'][()]))
            self.conv5[0].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv5.0.conv.weight'][()]))
            self.conv5[1].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv5.1.conv.bias'][()]))
            self.conv5[1].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv5.1.conv.weight'][()]))
            self.conv5[2].conv.bias = nn.Parameter(torch.from_numpy(params['rpn.features.conv5.2.conv.bias'][()]))
            self.conv5[2].conv.weight = nn.Parameter(torch.from_numpy(params['rpn.features.conv5.2.conv.weight'][()]))

        self.conv1 = nn.Sequential(*self.conv1)
        self.conv2 = nn.Sequential(*self.conv2)
        self.conv3 = nn.Sequential(*self.conv3)
        self.conv4 = nn.Sequential(*self.conv4)
        self.conv5 = nn.Sequential(*self.conv5)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
