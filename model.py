import math
import torch
import torch.nn as nn
import utils
import torch.utils.checkpoint as checkpoint
from torch.nn import Parameter
from torch.nn import functional as F


class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, subnet_width=1.0):
        super(MaskConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)
        self.subnet_width = subnet_width
        self.register_buffer("beta", torch.ones(self.weight.data.size(1), dtype=torch.long))
        self.register_buffer("active_index", torch.tensor(list(range(self.weight.data.size(1))), dtype=torch.int))
        self.student_flag = False

    def sample_beta(self, active_index):
        if len(active_index) < self.weight.data.size(1):
            self.beta.fill_(0)
            self.beta.index_fill_(0, torch.tensor(active_index).type_as(self.beta), 1)
        else:
            self.beta.fill_(1)

    def reset_beta(self):
        self.beta.fill_(1)

    def forward(self, input):
        active_index = self.beta.nonzero().squeeze()
        new_input = input.index_select(1, active_index)
        new_weight = self.weight.index_select(1, active_index)

        return F.conv2d(new_input, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv1x1(in_plane, out_plane, stride=1, subnet_width=1.0):
    """
    1x1 convolutional layer
    """
    return MaskConv2d(in_plane, out_plane, kernel_size=1, stride=stride, padding=0, bias=False, subnet_width=subnet_width)


def conv3x3(in_plane, out_plane, stride=1, subnet_width=1.0):
    "3x3 convolution with padding"
    return MaskConv2d(in_plane, out_plane, kernel_size=3, stride=stride,
                      padding=1, bias=False, subnet_width=subnet_width)


def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)


class MaskBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, subnet_width=1.0, drop_rate=0):
        super(MaskBasicBlock, self).__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=False)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, subnet_width=subnet_width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0
        self.n_paths = 2
        self.n_channels = [planes, 1]
        self.register_buffer("path_beta", torch.ones(self.n_paths, dtype=torch.long, requires_grad=False))
        self.alpha = Parameter(torch.ones(self.n_paths))

    def sample_beta(self, active_paths, active_indexes):
        # assert len(active_indexes)==2, 'num_active is not 2 in MaskBasicBlock'
        self.conv2.sample_beta(active_indexes[0])
        self.path_beta.fill_(0)
        self.path_beta.index_fill_(0, torch.tensor(active_paths).type_as(self.path_beta), 1)

    def reset_beta(self):
        self.conv2.reset_beta()
        self.path_beta.fill_(1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.drop_rate > 0:
            out = self.dropout(out)

        if self.path_beta.sum().item() < self.n_paths:
            out = out * self.path_beta[0]
            residual = residual * self.path_beta[1]

        out += residual
        out = self.relu(out)

        return out


class MaskBasicBlockCIFAR(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, subnet_width=1.0, drop_rate=0):
        super(MaskBasicBlockCIFAR, self).__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=False)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, subnet_width=subnet_width)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.n_paths = 2
        self.n_channels = [planes, 1]
        self.register_buffer("path_beta", torch.ones(self.n_paths, dtype=torch.long, requires_grad=False))
        self.alpha = Parameter(torch.ones(self.n_paths))

    def sample_beta(self, active_paths, active_indexes):
        self.conv2.sample_beta(active_indexes[0])
        self.path_beta.fill_(0)
        self.path_beta.index_fill_(0, torch.tensor(active_paths).type_as(self.path_beta), 1)

    def reset_beta(self):
        self.conv2.reset_beta()
        self.path_beta.fill_(1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.drop_rate > 0:
            out = self.dropout(out)

        if self.path_beta.sum().item() < self.n_paths:
            out = out * self.path_beta[0]
            x = x * self.path_beta[1]

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MaskBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, subnet_width=1.0, drop_rate=0):
        super(MaskBottleneck, self).__init__()
        self.name = "resnet-bottleneck"
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=False)
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, subnet_width=subnet_width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0
        self.n_paths = 2
        self.n_channels = [planes, planes, 1]
        self.register_buffer("path_beta", torch.ones(self.n_paths, dtype=torch.long, requires_grad=False))
        self.alpha = Parameter(torch.ones(self.n_paths))

    def sample_beta(self, active_paths, active_indexes):
        # assert len(active_indexes)==2, 'num_active is not 2 in MaskBasicBlock'
        self.conv2.sample_beta(active_indexes[0])
        self.path_beta.fill_(0)
        self.path_beta.index_fill_(0, torch.tensor(active_paths).type_as(self.path_beta), 1)

    def reset_beta(self):
        self.conv2.reset_beta()
        self.path_beta.fill_(1)

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

        if self.drop_rate > 0:
            out = self.dropout(out)

        if self.path_beta.sum().item() < self.n_paths:
            out = out * self.path_beta[0]
            residual = residual * self.path_beta[1]

        out += residual
        out = self.relu(out)
        return out


class MaskBottleneckCIFAR(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, subnet_width=1.0, drop_rate=0):
        super(MaskBottleneckCIFAR, self).__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=False)
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, subnet_width=subnet_width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, self.expansion * planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.n_paths = 2
        self.n_channels = [planes, planes, 1]
        self.register_buffer("path_beta", torch.ones(self.n_paths, dtype=torch.long, requires_grad=False))
        self.alpha = Parameter(torch.ones(self.n_paths))

    def sample_beta(self, active_paths, active_indexes):
        # assert len(active_indexes)==2, 'num_active is not 2 in MaskBasicBlock'
        self.conv2.sample_beta(active_indexes[0])
        self.path_beta.fill_(0)
        self.path_beta.index_fill_(0, torch.tensor(active_paths).type_as(self.path_beta), 1)

    def reset_beta(self):
        self.conv2.reset_beta()
        self.path_beta.fill_(1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.drop_rate > 0:
            out = self.dropout(out)

        if self.path_beta.sum().item() < self.n_paths:
            out = out * self.path_beta[0]
            x = x * self.path_beta[1]

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MaskPreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, subnet_width=1.0, drop_rate=0):
        super(MaskPreActBlock, self).__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, subnet_width=subnet_width)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride)
            )

        self.n_paths = 2
        self.n_channels = [planes, 1]
        self.register_buffer("path_beta", torch.ones(self.n_paths, dtype=torch.long, requires_grad=False))
        self.alpha = Parameter(torch.ones(self.n_paths))

    def sample_beta(self, active_paths, active_indexes):
        self.conv2.sample_beta(active_indexes[0])
        self.path_beta.fill_(0)
        self.path_beta.index_fill_(0, torch.tensor(active_paths).type_as(self.path_beta), 1)

    def reset_beta(self):
        self.conv2.reset_beta()
        self.path_beta.fill_(1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        if self.drop_rate > 0:
            out = self.dropout(out)

        if self.path_beta.sum().item() < self.n_paths:
            out = out * self.path_beta[0]
            shortcut = shortcut * self.path_beta[1]

        out += shortcut
        return out


class MaskPreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, subnet_width=1.0, drop_rate=0):
        super(MaskPreActBottleneck, self).__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv1x1(in_planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, subnet_width=subnet_width)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, self.expansion * planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride)
            )

        self.n_paths = 2
        self.n_channels = [planes, planes, 1]
        self.register_buffer("path_beta", torch.ones(self.n_paths, dtype=torch.long, requires_grad=False))
        self.alpha = Parameter(torch.ones(self.n_paths))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))

        if self.drop_rate > 0:
            out = self.dropout(out)

        if self.path_beta.sum().item() < self.n_paths:
            out = out * self.path_beta[0]
            shortcut = shortcut * self.path_beta[1]

        out += shortcut
        return out


class ResNet_CIFAR(nn.Module):
    def __init__(self, args, depth, num_classes=10, subnet_width=1.0, drop_rate=0, use_checkpoint=False):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 64
        self.subnet_width = subnet_width
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        assert depth in [18, 34, 50], 'invalid model depth'
        if depth == 18:
            num_blocks = [2, 2, 2, 2]
            block = MaskBasicBlockCIFAR
        elif depth == 34:
            num_blocks = [3, 4, 6, 3]
            block = MaskBasicBlockCIFAR
        elif depth == 50:
            num_blocks = [3, 4, 6, 3]
            block = MaskBottleneckCIFAR

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, subnet_width=subnet_width, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, subnet_width=subnet_width, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, subnet_width=subnet_width, drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, subnet_width=subnet_width, drop_rate=drop_rate)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.use_checkpoint = use_checkpoint

    def _make_layer(self, block, planes, num_blocks, stride, subnet_width=1.0, drop_rate=0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, subnet_width=subnet_width, drop_rate=drop_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.use_checkpoint:
            out = checkpoint.checkpoint(self.layer1, out)
            out = checkpoint.checkpoint(self.layer2, out)
            out = checkpoint.checkpoint(self.layer3, out)
            out = checkpoint.checkpoint(self.layer4, out)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PreActResNet_CIFAR(nn.Module):
    def __init__(self, depth, num_classes=10, subnet_width=1.0, drop_rate=0, use_checkpoint=False):
        super(PreActResNet_CIFAR, self).__init__()
        self.in_planes = 64
        self.subnet_width = subnet_width
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        assert depth in [18, 34, 50], 'invalid model depth'
        if depth == 18:
            num_blocks = [2, 2, 2, 2]
            block = MaskPreActBlock
        elif depth == 34:
            num_blocks = [3, 4, 6, 3]
            block = MaskPreActBlock
        elif depth == 50:
            num_blocks = [3, 4, 6, 3]
            block = MaskPreActBottleneck

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, subnet_width=subnet_width, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, subnet_width=subnet_width, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, subnet_width=subnet_width, drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, subnet_width=subnet_width, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.use_checkpoint = use_checkpoint

    def _make_layer(self, block, planes, num_blocks, stride, subnet_width=1.0, drop_rate=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, subnet_width=subnet_width, drop_rate=drop_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_checkpoint:
            out = checkpoint.checkpoint(self.layer1, out)
            out = checkpoint.checkpoint(self.layer2, out)
            out = checkpoint.checkpoint(self.layer3, out)
            out = checkpoint.checkpoint(self.layer4, out)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_IMAGENET(nn.Module):

    def __init__(self, depth, num_classes=1000, subnet_width=1.0, drop_rate=0, use_checkpoint=False):
        self.inplanes = 64
        super(ResNet_IMAGENET, self).__init__()
        self.num_classes = num_classes
        if depth < 50:
            block = MaskBasicBlock
        else:
            block = MaskBottleneck

        if depth == 18:
            layers = [2, 2, 2, 2]
        elif depth == 34:
            layers = [3, 4, 6, 3]
        elif depth == 50:
            layers = [3, 4, 6, 3]
        elif depth == 101:
            layers = [3, 4, 23, 3]
        elif depth == 152:
            layers = [3, 8, 36, 3]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], subnet_width=subnet_width, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, subnet_width=subnet_width,
                                       drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, subnet_width=subnet_width,
                                       drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, subnet_width=subnet_width,
                                       drop_rate=drop_rate)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.use_checkpoint = use_checkpoint

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, MaskConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, subnet_width=1.0, drop_rate=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, subnet_width=subnet_width, drop_rate=drop_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, subnet_width=subnet_width, drop_rate=drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.layer1, x)
            x = checkpoint.checkpoint(self.layer2, x)
            x = checkpoint.checkpoint(self.layer3, x)
            x = checkpoint.checkpoint(self.layer4, x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# wideresnet
class MaskWideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, subnet_width=1.0):
        super(MaskWideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes, subnet_width=subnet_width)

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and conv1x1(in_planes, out_planes, stride=stride) or None

        self.n_paths = 2
        self.n_channels = [out_planes, 1]
        self.register_buffer("path_beta", torch.ones(self.n_paths, dtype=torch.long, requires_grad=False))
        self.alpha = Parameter(torch.ones(self.n_paths))

    def sample_beta(self, active_paths, active_indexes):
        self.conv2.sample_beta(active_indexes[0])
        self.path_beta.fill_(0)
        self.path_beta.index_fill_(0, torch.tensor(active_paths).type_as(self.path_beta), 1)

    def reset_beta(self):
        self.conv2.reset_beta()
        self.path_beta.fill_(1)

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, subnet_width=1.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, subnet_width)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, subnet_width=1.0):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, subnet_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, subnet_width=1.0, use_checkpoint=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = MaskWideBasicBlock
        self.num_classes = num_classes
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, subnet_width)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, subnet_width)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, subnet_width)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, subnet_width)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.use_checkpoint = use_checkpoint

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.use_checkpoint:
            out = checkpoint.checkpoint(self.conv1, x)
            out = checkpoint.checkpoint(self.block1, out)
            out = checkpoint.checkpoint(self.block2, out)
            out = checkpoint.checkpoint(self.block3, out)
        else:
            out = self.conv1(x)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)