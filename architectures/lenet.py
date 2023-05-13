#!coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) #default: stride=1,padding=0
        self.conv2 = nn.Conv2d(6, 16,kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):#1*128*128
        out = F.relu(self.conv1(x))#6*124*124
        out = F.max_pool2d(out, 2)#6*62*62
        out = F.relu(self.conv2(out))#16*58*58
        out = F.max_pool2d(out, 2)#16*29*29
        out = F.relu(self.conv3(out))#32*25*25
        out = F.max_pool2d(out, 2)#32*12*12
        out = F.relu(self.conv4(out))#64*8*8
        out = F.max_pool2d(out, 2)#64*4*4
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return self.fc3(out)

def test():
    print('--- run lenet test ---')
    net = LeNet(10);
    print(net)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
