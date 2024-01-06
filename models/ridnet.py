import torch
import torch.nn as nn
import torch.nn.functional as F
from elementary_ops import *

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.gap = nn.AdaptiveMaxPool2d(1)

        self.c1 = BasicBloc(channels, channels//reduction, 1, 1, 0)
        self.c2 = BasicBloc(channels//reduction, channels, 1, 1, 0)

    def forward(self, x):
        y = self.gap(x)
        y = self.c1(y)
        y = self.c2(y)
        return x * y
    
class EAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EAM, self).__init__()

        self.r1 = MergeAndRunDual(in_channels, out_channels)
        self.r2 = ResidualBlock(in_channels, out_channels)
        self.r3 = EResidualBlock(in_channels, out_channels)
        self.ca = ChannelAttention(in_channels)

    def forward(self, x):
        r1 = self.r1(x)            
        r2 = self.r2(r1)       
        r3 = self.r3(r2)
        
        out = self.ca(r3)
        return out + x # short skip connection

class RIDNet(nn.Module):
    def __init__(self, 
                 in_channels, num_features):
        super(RIDNet, self).__init__()

        self.head = BasicBloc(in_channels, num_features, 3, 1, 1)

        self.eam1 = EAM(num_features, num_features)
        self.eam2 = EAM(num_features, num_features)
        self.eam3 = EAM(num_features, num_features)
        self.eam4 = EAM(num_features, num_features)

        self.tail = nn.Conv2d(num_features, in_channels, 3, 1, 1, 1)

    def forward(self, x):
        h = self.head(x)

        x_eam = self.eam1(h)
        x_eam = self.eam2(x_eam)
        x_eam = self.eam3(x_eam)
        x_eam = self.eam4(x_eam)

        x_lsc = x_eam + h #long skip connection

        x_out = self.tail(x_lsc)

        return x + x_out