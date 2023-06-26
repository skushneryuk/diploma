# Библиотеки для обучения
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import torch.nn.init as init


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, upsample_mode='bilinear'):
        super().__init__()
        if upsample:
            self.up = nn.Sequential([
                nn.Upsample(scale_factor=2, mode=upsample_mode),
                nn.Conv2d(in_channels, in_channels // 2, 1, bias=False)
            ])
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, out_channels)

    def forward(self, prev, skip):
        return self.conv(
            torch.cat([self.up(prev), skip], dim=-3)
        )


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth=3, start_channels=32,
                 upsample=False, upsample_mode='bilinear'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        self.multiple = 2 ** depth
        self.start_channels = start_channels
        self.threshold = 0.5

        self.upsample = upsample
        if upsample:
            self.upsample_mode = upsample_mode

        self.downs = [DoubleConv(n_channels, start_channels)]
        channels = (start_channels, start_channels * 2)
        for k in range(depth - 1):
            self.downs.append(DownBlock(*channels))
            channels = (channels[1], channels[1] * 2)

        self.downs = nn.ModuleList(self.downs)
        self.center = DownBlock(*channels)

        channels = (channels[1], channels[1] // 2)
        self.ups = []
        for k in range(depth):
            self.ups.append(UpBlock(*channels))
            channels = (channels[1], channels[1] // 2)
        self.ups = nn.ModuleList(self.ups)

        self.out = nn.Conv2d(channels[0], n_classes, 1)

        # initialize weights
        self.apply(weight_init)

    def set_threshold(self, ths):
        self.threshold = ths

    def forward(self, x):
        H, W = x.shape[-2:]
        if H % self.multiple != 0 or W % self.multiple != 0:
            new_H = ((H - 1) // self.multiple + 1) * self.multiple
            new_W = ((W - 1) // self.multiple + 1) * self.multiple
            x = F.pad(x, [0, new_W - W, 0, new_H - H])

        skips = []
        out = x
        for d_block in self.downs:
            out = d_block(out)
            skips.append(out)
        out = self.center(out)

        for skip, u_block in zip(skips[::-1], self.ups):
            out = u_block(out, skip)

        out = self.out(out)
        if H % self.multiple != 0 or W % self.multiple != 0:
            out = out[:, :, :H, :W]

        return out

    def predict(self, x):
        logits = self.forward(x)
        if self.n_classes != 1:
            return logits.argmax(dim=-3)
        return (F.sigmoid(logits) >= self.threshold).to(int)
