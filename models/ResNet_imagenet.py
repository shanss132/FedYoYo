"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


# from utils import load_state_dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils import autocast
use_fp16 = False

if use_fp16:
    from torch.cuda.amp import autocast
else:
    class Autocast(): # This is a dummy autocast class
        def __init__(self):
            pass
        def __enter__(self, *args, **kwargs):
            pass
        def __call__(self, arg=None):
            if arg is None:
                return self
            return arg
        def __exit__(self, *args, **kwargs):
            pass

    autocast = Autocast()
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# This class is from LDAM: https://github.com/kaidic/LDAM-DRW.
class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, dropout=None, num_classes=1000, use_norm=True, reduce_dimension=True, layer3_output_dim=None, layer4_output_dim=None, load_pretrained_weights=False, returns_feat=True, s=30):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 192
            else:
                layer3_output_dim = 256

        if layer4_output_dim is None:
            if reduce_dimension:
                layer4_output_dim = 384
            else:
                layer4_output_dim = 512

        self.layer3 = self._make_layer(block, layer3_output_dim, layers[2], stride=2)
        self.layer4 = self._make_layer(block, layer4_output_dim, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=2)
        
        self.use_dropout = True if dropout else False

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        if use_norm:
            self.linear = NormedLinear(layer4_output_dim * block.expansion, num_classes)
        else:
            s = 1
            self.linear = nn.Linear(layer4_output_dim * block.expansion, num_classes)

        self.returns_feat = returns_feat

        self.s = s

       

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        with autocast():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            
            x = x.view(x.size(0), -1)
            self.feat = x

            # if self.use_dropout:
                # x = self.dropout(x)

            x = self.linear(x)

            x = x * self.s # This hyperparam s is originally in the loss function, but we moved it here to prevent using s multiple times in distillation.
                
        # if self.returns_feat:
        return self.feat, x            
        # else:
        #     return x
if __name__=='__main__':
    # 指定 GPU 设备
    device = 'cuda'

    # 初始化模型并移动到指定设备
    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)

    # 创建输入并移动到指定设备
    input = torch.randn(128, 3, 244, 244).to(device)

    # 前向传播
    feature, out = model(input)

    # 打印输出尺寸
    print(feature.shape, out.shape)