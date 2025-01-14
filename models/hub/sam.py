import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================== Spatial Attention Map ===================================
# tìm ra pixel nào đóng góp sự quan trọng nhất
# output sẽ là một refined feature map
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels, affine=True, momentum=0.99, eps=1e-3) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class ChannelPooling(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        self.pool = ChannelPooling()
        self.conv = ConvLayer(2, 1, kernel_size, stride=1, padding=(kernel_size-1)//2, relu=False)
    
    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out)
        scale = torch.sigmoid(out)
        return scale * x