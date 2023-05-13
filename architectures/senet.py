#!coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride!=1 or in_planes!=planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers (Use nn.Conv2d instead of nn.Linear)
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        #w = F.sigmoid(self.fc2(w))  # 0.4.0
        w = torch.sigmoid(self.fc2(w)) # 0.4.1.post2
        # Excitation
        out = out * w

        out += self.shortcut(x)
        return F.relu(out)

class PreActBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride!=1 or in_planes!=planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, (out.size(2),))
        w = F.relu(self.fc1(w))
        #w = F.sigmoid(self.fc2(w))  # 0.4.0
        w = torch.sigmoid(self.fc2(w)) # 0.4.1.post2
        # Excitation
        out = out * w

        out += shortcut
        return out

class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=11, drop_ratio=0.5):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block,128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block,256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block,512, num_blocks[3], stride=2)
        self.fc1 = nn.Linear(2048, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride]+[1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):#1*128*128
        out = F.relu(self.bn1(self.conv1(x)))#64*128*128
        out = self.layer1(out)#64*64*64
        out = self.layer2(out)#128*32*32
        out = self.layer3(out)#256*16*16
        out = self.layer4(out)#512*8*8
        out = F.avg_pool2d(out, 4)#512*2*2
        out = out.view(out.size(0), -1)
        return self.fc1(out)

def SENet18(num_classes, drop_ratio):
    return SENet(PreActBlock, [2,2,2,2])

def test():
    print('--- run senet test ---')
    x = torch.randn(1,1,128,128)
    for net in [SENet18(11,0.5)]:
#         print(net)
        y = net(x)
        print(y.size())