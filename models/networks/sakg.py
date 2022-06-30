import jittor as jt
import jittor.nn as nn


class SAKG(nn.Module):
    def __init__(self, in_channels, out_channels=64, upscaling_factor=2, kernel_size=5):
        super().__init__()
        self.s = upscaling_factor
        self.k = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 1x1 conv
        self.feature_channel_compression = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=1
        )
        semantic_kernel_channels = self.k*self.k*self.s*self.s
        self.semantic_kernel_generation = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=semantic_kernel_channels,
            kernel_size=1
        )
        self.feature_shuffle = nn.PixelShuffle(upscale_factor=self.s)
        self.channelwise_normalization = nn.Softmax(dim=1)
        
    def execute(self, x):
        dx = self.feature_channel_compression(x)
        dx = self.semantic_kernel_generation(dx)
        dx = self.feature_shuffle(dx)
        dx = self.channelwise_normalization(dx)
        return dx
    
if __name__ == "__main__":
    jt.use_cuda = 1
    x = jt.rand((1,3,32,32))
    sakg = SAKG(x.shape[1])
    y = sakg(x)
    print(x.shape, y.shape)
    
    