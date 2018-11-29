import torch.nn as nn
import torch.nn.functional as F

import math

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F



model_url = {"vgg16": 'https://download.pytorch.org/models/vgg16-397923af.pth', 
            "vgg16_bn": 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'}
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

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
        all_alive = np.concatenate(alive_parameters)
        percentile_value = np.percentile(abs(all_alive), q)
        print(f"Pruning with Threshold: {percentile_value}")
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                module.prune(threshold=percentile_value)
    def prune_by_std(self, s=0.25):
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cupy().numpy()) * s
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
        self.weight_data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

        
class VGG(PruningModule):
    def __init__(self, features, num_classes=1000, init_weights=True, mask=False):
        super(VGG, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.features = features
        self.classifier = nn.Sequential(
            linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights(linear)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self, linear):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

if __name__ == "__main__":
    print("Testing")
    vgg = vgg16()
    print("Successfully Created")