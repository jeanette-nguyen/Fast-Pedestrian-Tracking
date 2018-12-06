import torch.nn as nn
import torch.nn.functional as F

import math

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module

class PruningModule(Module):
    def prune_by_percentile(self, q=5.0, **kwargs):
        """
        Prunes based off of percentile
        """
        al_parameters = []
        for name, p in self.named_parameters():
            # skip bias term for pruning
            if 'bias' in name or 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]
            al_parameters.append(alive)
        all_alive = np.concatenate(al_parameters)
        percentile_value = np.percentile(abs(all_alive), q)
        print(f"Pruning with Threshold: {percentile_value}")
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                module.prune(threshold=percentile_value)

    def prune_by_std(self, s=0.25, debug=False, batch_norm=False):
        for i, (name, module) in enumerate(self.named_modules()):
            if 'Masked' in str(module) and 'Sequential' not in str(module):
                if debug:
                    print("Pruning : ", str(name))
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                module.prune(threshold)


class MaskedLinear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_params()
    def reset_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_feaetures=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
    def prune(self, threshold):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

class MaskedConvolution(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False):
        super(MaskedConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.mask = Parameter(torch.ones([out_channels, in_channels, kernel_size, kernel_size]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_params()
    def reset_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input):
        return F.conv2d(input, self.weight * self.mask, self.bias, stride=self.stride, padding=self.padding)
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channel=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', k_size=' + str(self.k_size) \
            + ', stride=' + str(self.stride) \
            + ', bias=' + str(self.bias is not None) + ')'
    def prune(self, threshold):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)