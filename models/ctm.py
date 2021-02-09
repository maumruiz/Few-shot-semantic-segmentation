"""
CTM module for few shot segmentation
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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

class CTM(nn.Module):
    """
    CTM module for few shot segmentation

    Args:
        in_channels:
            number of input channels
    """
    def __init__(self, n_way, n_shot, n_queries, in_channels=3, pretrained_path=None):
        super().__init__()
        
        # TODO: Try other network combinations: Using resnet block, changing d2, d3, m2, m3

        in_concentrator = n_shot * in_channels
        self.concentrator = self._make_vgg_layer(2, in_concentrator, in_channels) # Input shape (NK, m1, d1, d1) -> Output shape (N, m2, d2, d2)

        in_projector = n_way * in_channels
        self.projector = self._make_vgg_layer(2, in_projector, in_channels) # Input shape (1, Nm2, d2, d2) -> Output shape (1, m3, d3, d3)

        self.reshaper = self._make_vgg_layer(2, in_channels, in_channels) # Output shape (NK, m3, d3, d3)

        self._init_weights()
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_queries = n_queries

    def forward(self, batch_supp_fts, batch_query_fts):
        # Support fts: Wa, Sh, B(1), C(512), H'(53), W'(53)
        # Query fts: Q, B(1), C(512), H'(53), W'(53)

        # Transform input to correct dimensions: supp(WaxSh, C, H, W) and query(Q, C, H, W)
        fts_size = batch_supp_fts.shape[-2:]
        supp_fts = batch_supp_fts[:, :, 0].view(-1, batch_supp_fts.size(3), *fts_size)
        query_fts = batch_query_fts[:, :, 0].view(-1, batch_query_fts.size(2), *fts_size)

        ### CONCENTRATOR
        # Reshape to have (n_way, n_shot*in_channels, H, W)
        supp_fts_reshape = supp_fts.view(self.n_way, -1, *fts_size)
        out_concentrator = self.concentrator(supp_fts_reshape)

        ### PROJECTOR
        # Reshape to have (1, n_way*in_channels, H, W)
        input_projector = out_concentrator.view(1, -1, *fts_size)
        out_projector = self.projector(input_projector)
        out_projector = F.softmax(out_projector, dim=1)
        
        ### ENHANCED FEATURES
        out_supp_fts = self.reshaper(supp_fts)
        out_supp_fts = torch.matmul(out_supp_fts, out_projector)

        out_query_fts = self.reshaper(query_fts)
        out_query_fts = torch.matmul(out_query_fts, out_projector)

        # Transform output to correct dimensions: supp(Wa, Sh, B, C, H, W) query(Q, B, C, H, W)
        out_supp_fts = out_supp_fts.view(self.n_way, self.n_shot, 1, -1, *fts_size)
        out_query_fts = out_query_fts.view(self.n_queries, 1, -1, *fts_size)

        return out_supp_fts, out_query_fts

    def _make_vgg_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _make_resnet_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        Make a resnet layer

        Args:
            blocks:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layers = []
        layers.append(Bottleneck(in_channels, out_channels))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
