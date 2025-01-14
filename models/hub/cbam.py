from .sam import SAM
from .cam import CAM
import torch.nn as nn
import torch

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, pool_types=['avg', 'max'], is_spatial=True):
        super(CBAM, self).__init__()
        self.cam = CAM(in_channels, reduction_ratio, pool_types)
        self.is_spatial = is_spatial
        if is_spatial:
            self.sam = SAM()
            
    def forward(self, x):
        out = self.cam(x)
        if self.is_spatial:
            out = self.sam(out)
        return out