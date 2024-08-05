# -*- coding: utf-8 -*-
# UFold Network
# https://github.com/uci-cbcl/UFold/blob/main/Network.py
import torch
import torch.nn as nn

CH_FOLD = 1


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


"""
ufold
data_seq: (batch_size, seq_len, 4)
data_lens: int
requires_channels: 17
"""


class UNet(nn.Module):
    def __init__(self, n_channels=17, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Maxpooling 2*2
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(
            in_channels=self.n_channels, out_channels=int(32 * CH_FOLD)
        )
        self.Conv2 = DoubleConv(
            in_channels=int(32 * CH_FOLD), out_channels=int(64 * CH_FOLD)
        )
        self.Conv3 = DoubleConv(
            in_channels=int(64 * CH_FOLD), out_channels=int(128 * CH_FOLD)
        )
        self.Conv4 = DoubleConv(
            in_channels=int(128 * CH_FOLD), out_channels=int(256 * CH_FOLD)
        )
        self.Conv5 = DoubleConv(
            in_channels=int(256 * CH_FOLD), out_channels=int(512 * CH_FOLD)
        )

        self.Up5 = UpConv(
            in_channels=int(512 * CH_FOLD), out_channels=int(256 * CH_FOLD)
        )
        self.Up_conv5 = DoubleConv(
            in_channels=int(512 * CH_FOLD), out_channels=int(256 * CH_FOLD)
        )

        self.Up4 = UpConv(
            in_channels=int(256 * CH_FOLD), out_channels=int(128 * CH_FOLD)
        )
        self.Up_conv4 = DoubleConv(
            in_channels=int(256 * CH_FOLD), out_channels=int(128 * CH_FOLD)
        )

        self.Up3 = UpConv(
            in_channels=int(128 * CH_FOLD), out_channels=int(64 * CH_FOLD)
        )
        self.Up_conv3 = DoubleConv(
            in_channels=int(128 * CH_FOLD), out_channels=int(64 * CH_FOLD)
        )

        self.Up2 = UpConv(in_channels=int(64 * CH_FOLD), out_channels=int(32 * CH_FOLD))
        self.Up_conv2 = DoubleConv(
            in_channels=int(64 * CH_FOLD), out_channels=int(32 * CH_FOLD)
        )

        self.Conv_1x1 = nn.Conv2d(
            int(32 * CH_FOLD), self.n_classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        # d1 = d1.squeeze(1)

        # make output matrix symmetric
        return torch.transpose(d1, -1, -2) * d1
