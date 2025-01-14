import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================== Channel Attention Map ===================================
# tìm ra channel nào đóng góp sự quan trọng nhất
# output sẽ là một refined feature map
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CAM(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CAM, self).__init__()
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            Flatten(),  # (b x c)
            nn.Linear(in_channels, in_channels //
                      reduction_ratio),  # (b x (c/r))
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),  # (b x (c/r))
            nn.Linear(in_channels // reduction_ratio, in_channels),  # (b x c)
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.pool_types = pool_types

    def forward(self, x):
        # x: (b x c x h x w)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # (b x c x 1 x 1)
                channel_att_raw = self.mlp(avg_pool) # (b x c)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))) # (b x c x 1 x 1)
                channel_att_raw = self.mlp(max_pool) # (b x c)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw # (b x c)
            else:
                channel_att_sum = channel_att_sum + channel_att_raw # (b x c)

        # print(channel_att_sum.unsqueeze(2).unsqueeze(3).shape) # (b x c x 1 x 1)
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x) # (b x c x h x w)
        return x * scale


if __name__ == '__main__':
    x = torch.rand(3, 4, 32, 32)
    y = torch.rand(3, 4, 1, 1)
    print(y)
    print(y.expand_as(x))
    # cam = ChannelGate(48)
    # out = cam(x)
    # print(out.shape)
    # fl = Flatten()
    # avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)),
    #                         stride=(x.size(2), x.size(3)))
    # print(avg_pool)
    # out = fl(avg_pool)
    # print(out)
