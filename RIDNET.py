
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 

def init_weights(modules):
    pass


class Merge_And_Run_Dual(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_And_Run_Dual, self).__init__()

        self.part1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.part2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 4, 4),
            nn.ReLU(inplace=True)
        )

        self.part3 = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        init_weights(self.modules)
        
    def forward(self, x):
        out1 = self.part1(x)
        out2 = self.part2(x)
        c = torch.cat([out1, out2], dim=1)

        out = self.part3(c)
        out = out + x
        return out
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(Merge_And_Run_Dual, self).__init__()

        self.part1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.part2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 4, 4),
            nn.ReLU(inplace=True)
        )

        self.part3 = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        init_weights(self.modules)
        
    def forward(self, x):
        out1 = self.part1(x)
        out2 = self.part2(x)
        c = torch.cat([out1, out2], dim=1)

        out = self.part3(c)
        out = out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d( in_channels, out_channels , 3, 1, 1),
            nn.ReLU( inplace= True ),
            nn.Conv2d( out_channels, out_channels, 3, 1, 1)
        )
        init_weights( self.modules )

    def forward( self, x):
        out = self.body(x)
        out = F.relu( out + x)
        return out

class EResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d( in_channels, out_channels, 3, 1, 1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        )

        init_weights( self.modules )

    def forward( self, x):
        out = self.body(x)
        out = F.relu( out + x)
        return out

class FeatureAttention(nn.Module):
    def __init__(self, channels, reduction = 16) -> None:
        super(FeatureAttention, self).__init__()
        
        self.gap      = nn.AdaptiveAvgPool2d(1)

        self.c1 = nn.Sequential(
            nn.Conv2d( channels, channels//reduction, 1, 1, 0),
            nn.ReLU( inplace= True )
        )
        
        self.c2 = nn.Sequential(
            nn.Conv2d( channels//reduction, channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward( self, x):
        gap = self.gap( x )
        x_out = self.c1(gap)
        x_out = self.c2(x_out)      
        return x * x_out

class EAM(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(EAM, self).__init__()

        # Merge-and-run unit ( Unite de fusion )
        self.r1 = Merge_And_Run_Dual(in_channels, out_channels)

        # Le bloc residuel
        self.r2 = ResidualBlock(in_channels, out_channels)

        # Le bloc residuel amelioré
        self.r3 = EResidualBlock(in_channels, out_channels)

        # Feature attention
        self.fa = FeatureAttention( in_channels)


    def forward(self, x):
        # Merge and run block
        x1 = self.r1(x)

        # Residual block
        x2 = self.r2(x1)

        # Enhance residual block
        x3  = self.r3(x2)

        # Feature attention
        out = self.fa(x3)

        return out + x # short skip connection


class RIDNet(nn.Module):
    def __init__(self, in_channels, num_features) -> None:
        super(RIDNet, self).__init__()

        kernel_size = 3

        self.head = nn.Sequential(
            nn.Conv2d( in_channels, num_features, kernel_size, stride=1, padding=1),
            nn.ReLU( inplace= True )
        )
        
        self.eam1 = EAM(in_channels=num_features, out_channels=num_features)
        self.eam2 = EAM(in_channels=num_features, out_channels=num_features)
        self.eam3 = EAM(in_channels=num_features, out_channels=num_features)
        self.eam4 = EAM(in_channels=num_features, out_channels=num_features)

        self.last_conv = nn.Conv2d(num_features,in_channels, kernel_size, stride=1, padding=1, dilation=1)

        self.init_weights()

    
    def forward(self, x):
        x1 = self.head(x) # feature extraction module

        x_eam = self.eam1(x1)                
        x_eam = self.eam2(x_eam)
        x_eam = self.eam3(x_eam)
        x_eam = self.eam4(x_eam)

        x_lsc = x_eam + x1 # Long skip connection

        x_out = self.last_conv(x_lsc) # reconstruction module

        x_out = x_out + x # Long skip connection

        return x_out
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MSE_loss(nn.Module):
    def __init__(self) -> None:
        super(MSE_loss).__init__()
    
    def forward(self, x, y):
        loss = F.mse_loss(x, y, reduction='mean')
        return loss * 100