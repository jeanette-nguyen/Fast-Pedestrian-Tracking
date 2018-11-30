import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torch
from PruningClasses import PruningModule, MaskedLinear, MaskedConvolution

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
    

def make_layers(cfg, in_channels=3, mask=False, batch_norm=False, bias=True):
    Conv2d = MaskedConvolution if mask else nn.Conv2d
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

model_url = {"vgg16": 'https://download.pytorch.org/models/vgg16-397923af.pth', 
            "vgg16_bn": 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'}
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(pretrained=False, 
          mask=False,
          in_channels=3,
          num_classes=1000,
          bias=True,
          debug=False,
          **kwargs):
    features = make_layers(cfg['D'], 
                           in_channels=in_channels, 
                           mask=mask, bias=bias)
    if pretrained:
        kwargs['init_weights'] = False
    if mask:
        kwargs['mask'] = True
    else:
        kwargs['mask'] = False
    model = VGG(features, num_classes=num_classes, **kwargs)
    if pretrained:
        pre_trained = model_zoo.load_url(model_url['vgg16'])
        new = list(pre_trained.items())
        curr_model_kvpair = model.state_dict()
        if debug:
            for k, v in curr_model_kvpair.items():
                print("curr :", str(k))
            for i in new:
                print("new :", str(i[0]))
        count = 0
        for k, v in curr_model_kvpair.items():
            if "mask" in k:
                continue
            layer_name, weights = new[count]
            curr_model_kvpair[k] = weights
            count += 1
        model.load_state_dict(curr_model_kvpair)
        #model.load_state_dict(model_zoo.load_url(model_url['vgg16']))
    return model

if __name__ == "__main__":
    print("Testing")
    vgg = vgg16()
    print("Successfully Created")