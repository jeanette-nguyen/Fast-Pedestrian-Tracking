import torch.nn as nn
import torch.nn.functional as F

import math

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.linear import Linear


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
        for i, (name, module) in enumerate(self.named_modules()):
            if 'Masked' in str(module) and 'Sequential' not in str(module):
                if debug:
                    print("Pruning : ", str(name))
                module.prune(threshold=percentile_value)

    def prune_by_std(self, s=0.25, debug=False, batch_norm=False):
        # if batch_norm:
        #     nmodules = ['features.0', 'features.3', 'features.7',
        #                 'features.10', 'features.14', 'features.17',
        #                 'features.20', 'features.24', 'features.27', 
        #                 'features.30', 'features.34', 'features.37',
        #                 'features.40', 'classifier.0', 'classifier.3',
        #                 'classifier.6']
        # else:
        #     nmodules = ['features.0', 'features.2', 'features.5', 
        #                 'features.7', 'features.10', 'features.12', 
        #                 'features.14', 'features.17', 'features.19', 
        #                 'features.21', 'features.24', 'features.26', 
        #                 'features.28', 'classifier.0', 'classifier.3',
        #                 'classifier.6']
        # for name, module in self.named_modules():
        #     '''
        #     Modules with weights are:
        #         [features.0, features.2, features.5, features.7, features.10, features.12, features.14, 
        #         features.17, features.19, features.21, features.24, features.26, features.28, classifier.0, classifier.3, classifier.6]
        #     '''
        #     if name in nmodules:
        #         threshold = np.std(module.weight.data.cupy().numpy()) * s
        #         print(f'Pruning with threshold : {threshold} for layer {name}')
        #         module.prune(threshold)
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
        self.sparse = False
        self.reset_params()

    def reset_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight * self.mask, self.bias)
        else:
            if self.sparse:
                return torch.mm(self.weight.data, input) + self.bias.view(self.weight.size(0), -1)
            else:
                return F.linear(input, self.weight, self.bias)
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
        if self.training:
            return F.conv2d(input, self.weight * self.mask, self.bias, stride=self.stride, padding=self.padding)
        else:
            return F.conv2d(input, self.weight, self.bias, stride=self.stride, padding=self.padding)
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

class SparseDenseLinear(Linear):
    def __init__(self, in_features=None, out_features=None, bias=True, Masked=None):
        if Masked is not None:
            b = True if Masked.bias is not None else None
            super(SparseDenseLinear, self).__init__(Masked.in_features,
                                                   Masked.out_features,
                                                   bias=b)
            self._sparse = Masked.sparse
            self.weight = Masked.weight
            self.bias = Masked.bias
        else:
            assert in_features is not None and out_features is not None, "Specify in_features and out_feautres if no MaskedLinear to initialize"
            super(SparseDenseLinear, self).__init__(in_features, out_features, bias=bias)
            self._sparse = False
            

    def _convert_to_dense(self):
        if str(torch.__version__) <= "0.4.1":
            if hasattr(self.weight, '_values') and hasattr(self.weight, '_indices'):
                self.weight.data = self.weight.coalesce().to_dense()
        else:
            if hasattr(self.weight, 'values') and hasattr(self.weight, 'indices'):
                self.weight.data = self.weight.coalesce().to_dense()
               
    def _convert_to_sparse(self):
        if str(torch.__version__) <= "0.4.1" and not hasattr(self.weight, '_values') and not hasattr(self.weight, '_indices'):
            print("Weight already sparse")
        elif str(torch.__version__) >= "1.0.0" and not hasattr(self.weight, "indicies") and not hasattr(self.weight, "values"):
            print("Weight already sparse")
        else:
            w = self.weight.data.cpu().numpy()
            matrix = coo_matrix(w)
            matrix = matrix.tocoo().astype(np.float32)
            ind = torch.from_numpy(np.vstack((matrix.row, matrix.col))).long()
            vals = torch.from_numpy(matrix.data)
            sh = torch.Size(matrix.shape)
            self.weight.data = torch.sparse.FloatTensor(ind, vals, sh)
    
    
    def forward(self, input):
        if self.sparse:
            insize = input.size()
            w_size = self.weight.size()
            if insize[1] == w_size[1]:
                return (torch.mm(self.weight, input.t()) + self.bias.view(self.out_features, -1)).t()
            else:
                return torch.mm(self.weight, input) + self.bias.view(self.out_features, -1)
        else:
            return F.linear(input, self.weight, self.bias)
    
    @property
    def sparse(self):
        return self._sparse
    
    @sparse.setter
    def sparse(self, value):
        if value and not self._sparse:
            self._sparse = True
            self._convert_to_sparse()
        elif not value and self._sparse:
            self._sparse = False
            self._convert_to_dense()
        else:
            print(f"Sparse already {value}")