import torch
import torch.nn as nn
import os


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN,self).__init__()
        self.layer1 = self.conv_sequential(3,16,3,1,2,2)
        self.layer2 = self.conv_sequential(16, 64, 3, 1, 2, 2)
        self.layer3 = self.conv_sequential(64, 128, 3, 1, 2, 2)
        self.layer4 = self.conv_sequential(128, 256, 3, 1, 2, 2)
        self.layer5 = self.conv_sequential(256, 512, 3, 1, 2, 2)
        self.transpose_layer2 = self.transpose_conv_sequential(num_classes,num_classes,4,2,1)
        self.transpose_layer8 = self.transpose_conv_sequential(num_classes,num_classes,16,8,4)
        self.ravel_layer32 = nn.Sequential(
            nn.Conv2d(512,num_classes,1),
            nn.ReLU(True)
        )
        self.ravel_layer16 = nn.Sequential(
            nn.Conv2d(256,num_classes,1),
            nn.ReLU(True)
        )
        self.ravel_layer8 = nn.Sequential(
            nn.Conv2d(128, num_classes, 1),
            nn.ReLU(True)
        )


    def forward(self,x):
        ret = self.layer1(x)
        ret = self.layer2(ret)
        ret = self.layer3(ret)
        x8 = ret
        ret = self.layer4(ret)
        x16 = ret
        ret = self.layer5(ret)
        x32 = ret
        x32 = self.ravel_layer32(x32)
        x16 = self.ravel_layer16(x16)
        x8 = self.ravel_layer8(x8)
        x32 = self.transpose_layer2(x32)
        x16 =x16+x32
        x16 = self.transpose_layer2(x16)
        x8 =x8+x16
        result = self.transpose_layer8(x8)
        return result


    def conv_sequential(self,in_size,out_size,kfilter,padding,kernel_size,stride):
        return nn.Sequential(
            nn.Conv2d(in_size,out_size,kfilter,padding=padding),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size,stride)
        )


    def transpose_conv_sequential(self,in_size,out_size,kfilter,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_size,out_size,kfilter,stride,padding,bias=False),
            nn.BatchNorm2d(out_size)
        )