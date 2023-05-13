#coding:utf-8
import torch
import torch.nn as nn

cfg = {
        'vgg11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def VGG11(num_classes):
    return VGG('vgg11', num_classes)

def VGG13(num_classes):
    return VGG('vgg13', num_classes)

def VGG16(num_classes):
    return VGG('vgg16', num_classes)

def VGG19(num_classes):
    return VGG('vgg19', num_classes)

class VGG(nn.Module):

    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.fc1(out)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def test():
    print('--- run vgg test ---')
    x = torch.randn(2,3,32,32)
    for net in [VGG11(10), VGG13(10), VGG16(10), VGG19(10)]:
        print(net)
        y = net(x)
        print(y.size())

