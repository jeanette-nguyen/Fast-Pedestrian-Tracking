import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Conv2d(nn.Module):
    """
    Class to define 2D convolution in our model - convolution may 
    be followed by batch normalization (if specified) and is
    always followed by ReLU.

    Attributes
    ----------
    conv:
        Convolution layer
    bn:
        Batch normalization layer or None
    relu:
        ReLU activation or None
    """
    def __init__(self, in_channels, out_channels, k, padding, bn=False, relu=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, 
                                 eps=0.001, 
                                 momentum=0, 
                                 affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

