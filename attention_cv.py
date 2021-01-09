import torch
from torch import nn
from torch.nn import functional as F


# reference: https://blog.paperspace.com/attention-mechanisms-in-computer-vision-cbam/
# original paper: https://arxiv.org/abs/1807.06521

# basic conv used in spatial gate.
# mapping the tensor got from ChannelPool to a 2-d matrix.
# then the matrix passes through the spatial gate to get attention weight
class BasicConv(nn.Module):
    # parameters:
    # in_chan: int, the number of channels of the inputs
    # out_chan: int, the number of channels of the outputs
    # kernel_size, stride, padding, dilation, groups, bias: parameters used in cnn
    # use_bn: bool, whether to use batch normalization or not
    # use_relu: bool, whether to use relu or not
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, use_relu=True, use_bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_chan = out_chan
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, 
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan,eps=1e-5, momentum=0.01, affine=True) if use_bn else None
        self.relu = nn.ReLU(inplace = True) if use_relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            self.relu(x)
        return x

# Channel Pool applies global max pooling and average pooling to the 
# feature tensors of shape (batch_size, channels, width, height) obtained 
# in Unet(in our case) and try to get a condense representation of our 
# feature tensors of shape (batch_size, 2, width, height)
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

    
# Spatial Gate combines BasicConv and ChannelPool, and applies 
# a sigmoid activation to get the attention weight. And finally applies the 
# attention wight to the original feature tensors through element-wise 
# multiplication as well as broadcasting. Intutively speaking, for each picture's
# feature tensor, we use only one channel, or one matrix of shape (width, height)
# to represent the feature tensor of shape (channels, width, height)
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, use_relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale
    
    
# Flatten will receive a feature tensor(of shape (batch_size, channels, 1, 1))
# and flattens it to a feature tensor of shape (batch_size, channels)
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Channel Gate first applies average pooling and max pooling to get two 
# feature matrix of shape (batch_size, channels), and add them together
# intutively speaking, for every picture's feature tensor in the batch, 
# we use only one number to represent a channel, and then applies a sigmoid
# to get the attention weight for every channels. Finally, applies the attention
# weight to original feature tensors through element-wise multiplication as well 
# as broadcasting.

# One important thing to notice is that it uses a Multi-layer Perceptron,
# first transforms the feature vector to a reduced space and then transforms
# it back, such approach was not optimal, but constrained to computational 
# cost, it was a necessary sacrifice. Notice there is a new model that uses 
# different approach, called ECA-Net: https://arxiv.org/abs/1910.03151
class ChannelGate(nn.Module):
    # parameters:
    # gate_channels: int, the number of channels of the inputs
    # reduction_ratio: int, used to determine the dimension of the hidden layer of the mlp
    # pool_types: list, pooling types that is used to get the feature vector.
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
    
# putting all together, CBAM incorperates Channel Gate and Spatial Gate
# receives a feature tensor of shape (batch_size, channels, width, height),
# and output a feature tensor of shape (batch_size, channels, width, height)
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out