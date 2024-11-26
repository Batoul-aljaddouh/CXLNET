import torch
import torchvision
import torch.nn.functional as F

import pandas as pd
import numpy as np


class Block(torch.nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, max_pool_kernel_size, batch_norm=False):
        super().__init__()


        self.max_pool_kernel_size=max_pool_kernel_size
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, padding=1)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(mid_channel)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            self.bn3 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = torch.nn.functional.relu(x, inplace=True)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = torch.nn.functional.relu(x, inplace=True)

        if self.max_pool_kernel_size!=1:
            x = torch.nn.functional.max_pool2d(x, kernel_size=self.max_pool_kernel_size)
        if self.batch_norm:
            x = self.bn3(x)
        out = x
        return out

 


class CxlNet(torch.nn.Module):
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)

    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)


    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode="nearest",image_size=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode
        self.image_size=image_size
        self.enc1 = Block(in_channels, 32, 64,2, batch_norm)
        self.enc2 = Block(64, 64, 64, 2, batch_norm)
        self.enc3 = Block(64, 128, 128, 2, batch_norm)
        self.enc4 = Block(128, 256, 256, 2, batch_norm)
        #self.enc3 = Block(256, 128, 128, 2, batch_norm)
        #self.enc4 = Block(128, 64, 64, 2, batch_norm)

        self.dec3 = Block(512, 256, 256, 1, batch_norm)
        self.dec2 = Block(256, 128, 128, 1, batch_norm)
        self.dec1 = Block(128, 64, 64, 1, batch_norm)
        self.dec0 = Block(64, 32, out_channels, 1, batch_norm)
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        outOfDec3 = self.dec3(torch.cat([enc1,
                                    self.up(enc2, enc1.size()[-2:]),
                                    self.up(enc3, enc1.size()[-2:]),
                                    self.up(enc4, enc1.size()[-2:]),
        ], 1))

        outOfDec2 = self.dec2(self.up(outOfDec3, (self.image_size,self.image_size)))
        outOfDec1 = self.dec1(outOfDec2)
        outOfDec0 = self.dec0(outOfDec1)
        return outOfDec0