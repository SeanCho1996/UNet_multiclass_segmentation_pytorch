import torch.nn as nn
import torch
from collections import defaultdict


# define model
down_feature = defaultdict(list)
filter_list = [i for i in range(6, 9)]


class down_sampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down_sampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)


    def forward(self, in_feat):
        x = self.conv(in_feat)
        down_feature[in_feat.device.index].append(x)
        x = self.pool(x)

        return x


class up_sampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(up_sampling, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        self.relu_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )


    def forward(self, in_feat):
        x = self.up_conv(in_feat)
        down_map = down_feature[in_feat.device.index].pop()
        x = torch.cat([x, down_map], dim=1)
        x = self.relu_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(UNet, self).__init__()
        self.input_conv = down_sampling(input_channels, 64)
        self.down_list = [down_sampling(2 ** i, 2 ** (i + 1)) for i in filter_list]
        self.down = nn.Sequential(*self.down_list)

        self.last_layer = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.up_init = up_sampling(1024, 512)
        self.up_list = [up_sampling(2 ** (i + 1), 2 ** i) for i in filter_list[::-1]]
        self.up = nn.Sequential(*self.up_list)

        self.output = nn.Conv2d(64, num_classes, 1)
        

    def forward(self, in_feat):
        x = self.input_conv(in_feat)
        x = self.down(x)
        x = self.last_layer(x)
        x = self.up_init(x)
        x = self.up(x)
        x = self.output(x)
        # out = self.segment(x)
        # return out
        return x