# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class DoubleConv(nn.Module):
    """(convolution=> ReLU) * 2"""

    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,  kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class subnet(nn.Module):

    def __init__(self, in_channels, n_classes):
        super(subnet, self).__init__()

        self.in_channels = in_channels
        self.n_classes=n_classes

        # downsampling
        self.conv1 = DoubleConv(self.in_channels, 64)#512,768
        self.maxpool1 = nn.MaxPool2d(kernel_size=4)#128,192

        self.conv2 = DoubleConv(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4)#32,48

        self.conv3 = DoubleConv(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)#16,24

        self.conv4 = DoubleConv(256, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)#8,12

        self.conv5 = DoubleConv(512, 1024)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))

        self.classifier=nn.Conv2d(1024,n_classes,kernel_size=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        conv5 = self.conv5(maxpool4)
        avgpool1=self.avgpool(conv5)

        x=self.classifier(avgpool1)

        cam=nn.functional.conv2d(conv5,self.classifier.weight)

        x = torch.squeeze(x,2)
        x = torch.squeeze(x,2)

        return x,cam
class MDN(nn.Module):

    def __init__(self, in_channels, n_classes):
        super(MDN, self).__init__()
        self.teachernet=subnet(in_channels, n_classes)
        self.studentnet=subnet(in_channels, n_classes)

    def forward(self, input_student,input_teacher):
        x_student,cam_student=self.studentnet(input_student)
        x_teacher, cam_teacher = self.teachernet(input_teacher)

        return x_student,x_teacher,cam_student,cam_teacher