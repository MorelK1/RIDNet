import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(modules):
    pass

# un bloc basic de convolution et d'activation relu
class BasicBloc(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, padding=1):
        super(BasicBloc, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, padding),
            nn.ReLU(inplace=True)
        )

        init_weights( self.modules )

    def forward( self, x ):
        return self.body(x)
    
class BasicBlocSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, padding=1):
        super(BasicBloc, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, padding),
            nn.Sigmoid()
        )

        init_weights( self.modules )

    def forward( self, x ):
        return self.body(x)

# l'unite de fusion (Merge and Run Module)
class MergeAndRunDual(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(MergeAndRunDual, self).__init__()

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
    


# le bloc residuel (Residual block)
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



# le bloc residuel ameliore (Enhancement Residual Block)
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