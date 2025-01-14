import torch
import torch.nn as nn
from .cbam import CBAM

def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, affine=True, momentum=0.99, eps=1e-3),
        nn.ReLU(inplace=True)
    )

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding, dilation=dilation, groups=in_channels,
                                   bias=bias)
        self.bnd = nn.BatchNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride, 0, 1, 1, bias=bias)
        self.bnp = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.relu(self.bnd(out))
        out = self.pointwise(out)
        out = self.relu(self.bnp(out))
        return out
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, keep_dim=False):
        super(Block, self).__init__()
        
        self.keep_dim = keep_dim
        stride_sep_conv2 = 2
        if keep_dim:
            stride_sep_conv2 = 1
        self.conv = conv3x3(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sep_conv1 = SeparableConv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=1)
        self.sep_conv2 = SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=stride_sep_conv2, bias=False, padding=1)
        self.cbam = CBAM(out_channels)
        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.conv(x)
        if not self.keep_dim:
            residual = self.maxp(residual)
        
        out = self.sep_conv1(x)
        out = self.sep_conv2(out)
        out = self.cbam(out)
        out += residual 
        # out = self.relu(out)
        return out
# class 
    
class Model(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 1
        self.conv1 = conv3x3(in_channels, 8)
        self.conv2 = conv3x3(8, 8)
        self.cbam = CBAM(8,8)
        
        # 2
        self.block1 = Block(8, 16)
        
        # 3
        self.block2 = Block(16, 32)
        
        # 4
        self.block3 = Block(32, 64)
        
        # 5
        self.block4 = Block(64, 128)
        
        #6 
        # self.block5 = Block(128, 128)
        
        # last conv to down to num_classes 
        self.last_conv = conv3x3(128, num_classes)
        
        # global avg-p
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # 1
        out = self.conv2(self.conv1(x))
        out = self.cbam(out)
        # out = self.relu(out)
        # print('1', out.shape)
        # 2
        out = self.block1(out)
        # print('2', out.shape)
        # 3
        out = self.block2(out)
        # print('3', out.shape)
        # 4
        out = self.block3(out)
        # print('4', out.shape)
        # 5
        out = self.block4(out)
        # print('5555555555', out.shape)
        # 6
        # out = self.block5(out)
        out = self.last_conv(out)
        out = self.avgp(out)
        out = out.view((out.shape[0], -1))
        return out   
        
if __name__ == "__main__":
    import time
    import numpy as np
    model = Model(1, 7)
    print('total params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()
    x = torch.rand(1, 1, 48, 48)
    t = []
    for i in range(10):
        t0 = time.time()
        out = model(x)
        t.append(time.time() - t0)
    t = np.array(t)
    print('time inference: ', np.mean(t))
   