import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dropout_block import *

class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet_s1, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=1)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=1,
                        stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(53*53*256, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.drop=nn.Dropout(0.3)
        self.drop1=DropBlock2D(block_size=8, drop_prob=0.3)
        self.drop2=DropBlock2D(block_size=12, drop_prob=0.3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.drop1(out)
        out = self.layer2(out)
        out = self.drop2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn5(self.conv5(out)))
        out = out.view(out.size(0), -1)
        out = self.fc1 (out)
        out = self.drop(out)
        out = self.fc2 (out)
        return out

class ResNet_s2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet_s2, self).__init__()
        self.in_planes =16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=1,
                        stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(320000, 1)
        # self.fc2 = nn.Linear(512, 1)
        self.drop1=DropBlock2D(block_size=8, drop_prob=0.3)
        self.drop2=DropBlock2D(block_size=12, drop_prob=0.3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.drop1(out)
        out = self.layer2(out)
        out = self.drop2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn5(self.conv5(out)))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        # out = self.fc2(out)
        return out

class ResNet_s3(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet_s3, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,
                        stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,
                        stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3,
                stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3,
                stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3,
                stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3,
                stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=3,
                stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(256*26*26, 1024)
        self.fc2 = nn.Linear(1024, 2)
        self.drop=nn.Dropout(0.3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        # out = F.relu(self.bn7(self.conv7(out)))
        # out = F.relu(self.bn8(self.conv8(out)))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

class ResNet_s4(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet_s4, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,
                        stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,
                        stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3,
                stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3,
                stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3,
                stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3,
                stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3,
                stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(7*7*512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = F.relu(self.bn7(self.conv7(out)))
        out = F.relu(self.bn8(self.conv8(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def model_1():
    return ResNet_s1(Bottleneck, [1,2,2,1])
def model_2():
    return ResNet_s2(Bottleneck, [2,3,3,1])
def model_3():
    return ResNet_s3(BasicBlock, [1,1,1,1])
def model_4():
    return ResNet_s4(BasicBlock, [1,1,1,1])