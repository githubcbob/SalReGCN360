import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d
from global_reasoning_unit import GloRe_Unit_2D_self, GloRe_Unit_2D_cross


def load_model(name='MSCG-Rx50', classes=7, node_size=(32,32)):
    if name == 'MSCG-Rx50':
        net = rx50_gcn_3head_4channel(out_channels=classes)
    elif name == 'MSCG-Rx101':
        net = rx101_gcn_3head_4channel(out_channels=classes)
    else:
        print('not found the net')
        return -1

    return net


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class rx50_gcn_3head_4channel(nn.Module):
    def __init__(self, out_channels=7, pretrained=True,
                 nodes=(32, 32), dropout=0,
                 enhance_diag=True, aux_pred=True):
        super(rx50_gcn_3head_4channel, self).__init__()  # same with  res_fdcs_v5

        self.aux_pred = aux_pred
        self.node_size = nodes
        self.num_cluster = out_channels

        resnet = se_resnext50_32x4d()
        self.layer0, self.layer1, self.layer2 = \
            resnet.layer0, resnet.layer1, resnet.layer2

        self.layer3 = resnet.layer3

        self.GloRe_self1 = GloRe_Unit_2D_self(1024, 512, True)
        self.GloRe_cross1 = GloRe_Unit_2D_cross(1024, 512, True)

        self.GloRe_self2 = GloRe_Unit_2D_self(512, 256, True)


        self.decoder1 = nn.Sequential(
            BasicConv2d(1024, 512, 3, padding=1, dilation=1),
            BasicConv2d(512, 512, 3, padding=1, dilation=1),
            BasicConv2d(512, 512, 3, padding=1, dilation=1)
        )

        self.decoder2 = nn.Sequential(
            BasicConv2d(512, 128, 3, padding=1, dilation=1),
            BasicConv2d(128, 128, 3, padding=1, dilation=1),
        )


        self.upsample = nn.Upsample(scale_factor=2)

        self.conv = nn.Conv2d(128, 1,
                              kernel_size=3, stride=1,
                              padding=1, dilation=1, bias=False)


        self.weight_xavier_init(self.decoder2, self.conv)
 
    def weight_xavier_init(*models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                    elif isinstance(module, nn.BatchNorm2d):
                        module.weight.data.fill_(1)
                        module.bias.data.zero_()


    def forward(self, x1, x2):

        x_size = x1.size()

        gx1 = self.layer0(x1)
        gx1 = self.layer1(gx1)
        gx1 = self.layer2(gx1)
        gx1 = self.layer3(gx1)

        gx2 = self.layer0(x2)
        gx2 = self.layer1(gx2)
        gx2 = self.layer2(gx2)
        gx2 = self.layer3(gx2)

        gx1_self = self.GloRe_self1(gx1)
        gx2_self = self.GloRe_self1(gx2)

        gx1_cross, gx2_cross = self.GloRe_cross1(gx1, gx2)

        gx1 = gx1_self + gx1_cross
        gx2 = gx2_self + gx2_cross

        gx1 = self.decoder1(gx1)
        gx1 = self.upsample(gx1)
        gx2 = self.decoder1(gx2)
        gx2 = self.upsample(gx2)
        
        gx1 = self.GloRe_self2(gx1)
        gx2 = self.GloRe_self2(gx2)

        gx1 = self.decoder2(gx1)
        gx1 = self.upsample(gx1)
        gx2 = self.decoder2(gx2)
        gx2 = self.upsample(gx2)
        gx1 = self.conv(gx1)
        gx2 = self.conv(gx2)

        gx1 = F.interpolate(gx1, x_size[2:], mode='bilinear', align_corners=False)
        gx2 = F.interpolate(gx2, x_size[2:], mode='bilinear', align_corners=False)

        return gx1, gx2





