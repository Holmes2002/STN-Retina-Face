#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:18:36 2019

@author: xingyu
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: shape (batch size, channel, height, width)
        Returns:
            x: shape (batch size, channel, 1, output_size)
        '''
        shape_x = x.shape
        if(shape_x[-1] < self.output_size[-1]):
            paddzero = torch.zeros((shape_x[0], shape_x[1], shape_x[2], self.output_size[-1] - shape_x[-1]))
            paddzero = paddzero.to('cuda:0')
            x = torch.cat((x, paddzero), axis=-1)

        stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
        avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = avg(x)
        return x


# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/AffineGridGenerator.cpp
class Advanced_STNet(nn.Module):
    
    def __init__(self):
        super(Advanced_STNet, self).__init__()

        # Spatial transformer localization-network
        # self.localization = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Increased filters to 32
        #     nn.BatchNorm2d(32),  # Added Batch Normalization
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),

        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Increased filters to 64
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),

        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Added an additional Conv layer
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # Replaced fixed pooling with adaptive pooling

        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Added an additional Conv layer
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # Replaced fixed pooling with adaptive pooling
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Added an additional Conv layer
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # Replaced fixed pooling with adaptive pooling

        # )
        self.localization = nn.Sequential(
            ResidualBlock(1, 32),    # Replace Conv2d(1, 32) with ResidualBlock
            nn.MaxPool2d(2, stride=2),

            ResidualBlock(32, 64),   # Replace Conv2d(32, 64) with ResidualBlock
            nn.MaxPool2d(2, stride=2),

            ResidualBlock(64, 128),  # Replace Conv2d(64, 128) with ResidualBlock
            nn.MaxPool2d(2, stride=2),

            ResidualBlock(128, 128),  # Replace Conv2d(128, 128) with ResidualBlock
            nn.MaxPool2d(2, stride=2),

            ResidualBlock(128, 256),  # Replace Conv2d(128, 256) with ResidualBlock
            nn.MaxPool2d(2, stride=2)
        )


        self.fc_loc = nn.Sequential(
            nn.Linear(256 * 10 * 10, 32),  # Adjust input size based on modified localization network
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    def affine_grid(self, theta, size, align_corners=False):
        N, C, H, W = size
        grid = self.create_grid(N, C, H, W).to('cuda')
        grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
        grid = grid.view(N, H, W, 2)
        return grid

    def create_grid(self, N, C, H, W):
        grid = torch.empty((N, H, W, 3), dtype=torch.float32)
        grid.select(-1, 0).copy_(self.linspace_from_neg_one(W))
        grid.select(-1, 1).copy_(self.linspace_from_neg_one(H).unsqueeze_(-1))
        grid.select(-1, 2).fill_(1)
        return grid
        
    def linspace_from_neg_one(self, num_steps, dtype=torch.float32):
        r = torch.linspace(-1, 1, num_steps, dtype=torch.float32)
        r = r * (num_steps - 1) / num_steps
        return r
    def forward(self, x):
        # Perform the localization network forward pass
        xs = self.localization(x)
        xs = xs.view(-1, 256 * 10 * 10)  # Flatten the output
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = self.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid)
        
        return x, theta
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Residual_STNet(nn.Module):
    def __init__(self):
        super(Residual_STNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
                ResidualBlock(1, 32),
                nn.MaxPool2d(2, stride=2),
                ResidualBlock(32, 64),
                nn.MaxPool2d(2, stride=2),
                ResidualBlock(64, 64),
                AdaptiveAvgPool2dCustom((8, 8))
            )
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 64),  # Adjust input size based on modified localization network
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def affine_grid(self, theta, size, align_corners=False):
        N, C, H, W = size
        grid = self.create_grid(N, C, H, W)
        grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
        grid = grid.view(N, H, W, 2)
        return grid

    def create_grid(self, N, C, H, W):
        grid = torch.empty((N, H, W, 3), dtype=torch.float32)
        grid.select(-1, 0).copy_(self.linspace_from_neg_one(W))
        grid.select(-1, 1).copy_(self.linspace_from_neg_one(H).unsqueeze_(-1))
        grid.select(-1, 2).fill_(1)
        return grid
        
    def linspace_from_neg_one(self, num_steps, dtype=torch.float32):
        r = torch.linspace(-1, 1, num_steps, dtype=torch.float32)
        r = r * (num_steps - 1) / num_steps
        return r

    def forward(self, x):
        # Perform the localization network forward pass
        xs = self.localization(x)
        xs = xs.view(-1, 64 * 8 * 8)  # Flatten the output
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # Generate the affine grid and apply the transformation
        grid = self.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid)
        
        return x, theta
def conv3x3_block(in_channels, out_channels, stride=1):
    n = 3 * 3 * out_channels
    w = math.sqrt(2.0 / n)
    conv_layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True,  # PyTorch does not use `bias_attr`; you can set `bias=False` if needed
    )

    # Initialize weights and bias manually
    nn.init.normal_(conv_layer.weight, mean=0.0, std=w)
    nn.init.constant_(conv_layer.bias, 0)
    
    block = nn.Sequential(conv_layer, nn.BatchNorm2d(out_channels), nn.ReLU())
    return block


class STN_Paddlde(nn.Module):
    def __init__(self, in_channels = 1,):
        super(STN_Paddlde, self).__init__()
        self.in_channels = in_channels
        self.stn_convnet = nn.Sequential(
            conv3x3_block(in_channels, 32),  # 32x64
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(32, 64),  # 16x32
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(64, 128),  # 8*16
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(128, 128),  # 4*8
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(128, 128),  # 2*4,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # 1*2
        self.stn_fc1 = nn.Sequential(
            nn.Linear(
                128* 10 *10 ,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                6,
            )
        )
        self.stn_fc1[-1].weight.data.zero_()
        self.stn_fc1[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    def affine_grid(self, theta, size, align_corners=False):
        N, C, H, W = size
        grid = self.create_grid(N, C, H, W)
        grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
        grid = grid.view(N, H, W, 2)
        return grid

    def create_grid(self, N, C, H, W):
        grid = torch.empty((N, H, W, 3), dtype=torch.float32)
        grid.select(-1, 0).copy_(self.linspace_from_neg_one(W))
        grid.select(-1, 1).copy_(self.linspace_from_neg_one(H).unsqueeze_(-1))
        grid.select(-1, 2).fill_(1)
        return grid
        
    def linspace_from_neg_one(self, num_steps, dtype=torch.float32):
        r = torch.linspace(-1, 1, num_steps, dtype=torch.float32)
        r = r * (num_steps - 1) / num_steps
        return r

    def forward(self, x):
        xs = self.stn_convnet(x)
        xs = xs.view(-1, 128 * 10 * 10)  # Flatten the output
        theta = self.stn_fc1(xs)
        theta = theta.view(-1, 2, 3)
        # Generate the affine grid and apply the transformation
        grid = self.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid)
        
        return x, theta
        return img_feat, x


if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = STN_Paddlde().to(device)
    # model.load_state_dict(torch.load('/home/data2/congvu/checkpoint/21_258.pt'))
    input = torch.Tensor(2, 1, 320, 320).to(device)
    output = model(input)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    print('output shape is', output.shape)
    assert False
    torch.onnx.export(model,               # model being run
                      input,                         # model input (or a tuple for multiple inputs)
                      'model.onnx',   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=17,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['bboxes'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size', 2: 'height', 3:'width'},    # variable length axes
                                    'bboxes' : [0, 1]})