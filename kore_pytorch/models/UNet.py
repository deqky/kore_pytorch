from collections import OrderedDict

import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.name = 'UNet'
        
        self.up_convs = nn.ModuleList([self.double_conv(3, 64)] + [self.double_conv(2 ** (i-1), 2**(i)) for i in range(7, 11)])
        self.down_convs = nn.ModuleList([self.double_conv(2 ** (i + 1), 2**(i)) for i in range(9, 5, -1)])
        self.upsample = nn.ModuleList([nn.ConvTranspose2d(2 ** (i + 1), 2**(i), kernel_size=2, stride=2) for i in range(9, 5, -1)])

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size = 1)
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )
        
    def forward(self, x):
        result = OrderedDict()
        upconvs = []
        for i in self.up_convs:
            x = i(x)
            upconvs.append(x)
            x = self.pool(x)
        upconvs.reverse()

        x = upconvs.pop(0)
        for upconv, upsample, downconv in zip(upconvs, self.upsample, self.down_convs):
            x = upsample(x)
            x = torch.cat([x, upconv], 1)
            x = downconv(x)
        
        result["out"] = self.classifier(x)
        return result
