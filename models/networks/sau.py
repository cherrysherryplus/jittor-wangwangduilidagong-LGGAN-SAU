import jittor as jt
import jittor.nn as nn

from models.networks.safu import SAFU
from models.networks.sakg import SAKG


import jittor as jt
import jittor.nn as nn

from models.networks.safu import SAFU
from models.networks.sakg import SAKG


class SAU(nn.Module):
    def __init__(self, in_channels, mid_channels=64, upscaling_factor=2, kernel_size=5, downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.k = kernel_size
        self.s = upscaling_factor
        if downsample:
            self.downsample = nn.Conv2d(
                in_channels = self.in_channels,
                out_channels = self.in_channels//2,
                kernel_size = 1
        )
        self.sakg_block = SAKG(
            in_channels = self.in_channels, 
            out_channels = mid_channels, 
            upscaling_factor = self.s,
            kernel_size= self.k
        )
        self.safu_block = SAFU(
            in_channels=self.in_channels,
            upscaling_factor=self.s,
            patch_size=self.k
        )
        
    def execute(self, x):
        b,c,h,w = x.shape
        # weight新增一个维度，方便点积
        dx = self.safu_block(x)
        weight = self.sakg_block(x).unsqueeze(1)
        # print(dx.shape)
        # print(weight.shape)

        # dim=1变为dim=2
        y = (dx*weight).sum(dim=2).view(b, c, h*self.s, w*self.s)
        if hasattr(self, 'downsample'):
            y = self.downsample(y)
        return y
    
    
if __name__ == "__main__":
    jt.use_cuda = 1
    x = jt.rand((2,3,32,32))
    sau = SAU(x.shape[1])
    y = sau(x)
    print(x.shape, y.shape)
    
