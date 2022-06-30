import jittor as jt
import jittor.nn as nn


Unfold = jt.make_module(nn.unfold)

class SAFU(nn.Module):
    def __init__(self, in_channels, upscaling_factor=2, patch_size=5):
        super().__init__()
        self.s = upscaling_factor
        self.k = patch_size
        self.feature_spatial_expansion = nn.Upsample(scale_factor=self.s, mode='nearest')
        kw, kh = patch_size, patch_size
        pw, ph = kw//2, kh//2
        self.in_channels = in_channels
        self.out_channels = in_channels*self.k*self.k
        self.sliding_local_block_extraction = Unfold(kernel_size=(kw,kh), padding=(pw,ph))
    
    def execute(self, x):
        b,_,h,w = x.shape
        dx = self.feature_spatial_expansion(x)
        dx = self.sliding_local_block_extraction(dx).view(b, self.in_channels, self.k*self.k, h*self.s, w*self.s)
        return dx
    
if __name__ == "__main__":
    jt.use_cuda = 1
    x = jt.rand((2,3,32,32))
    safu = SAFU(x.shape[1])
    y = safu(x)
    print(x.shape, y.shape)
    
    