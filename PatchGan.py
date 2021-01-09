import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from attention_cv import CBAM
from util import *

# original paper: https://arxiv.org/abs/1611.07004
    
# contract block: uses two cnns, double the channels
# at each block, and followed by a maxpool
# thought: injecting a noise into such a block,
# like what styleGAN did
class ContractingBlock(nn.Module):
    # parameters:
    # in_chan: int, the number of channels of the inputs
    # use_dp: bool, whether to use dropout or not
    # use_bn: bool, whether to use batch normalization or not
    # use_att: bool, whether to use attention or not
    def __init__(self, input_channels, use_dropout=False, use_bn=True, use_att=True):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels * 2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout
        self.use_att = use_att
        if use_att:
            self.cbam = CBAM(gate_channels = input_channels*2)
    # parameter:
    # x: torch.tensor, input tensor of shape (batch_size, in_chann, height, width)
    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        if self.use_att:
            x = self.cbam(x)
        return x

# expand block: similar structure to contract block, 
# but first upsampling the input tensor by doubling its size,
# applies a cnn and concatenate it with a tensor, similar to resnet
# and then the same as contract block, two cnn,
# also, optional dropout and batch normalization
class ExpandingBlock(nn.Module):
    # parameters:
    # in_chan: int, the number of channels of the inputs
    # use_dp: bool, whether to use dropout or not
    # use_bn: bool, whether to use batch normalization or not
    # use_att: bool, whether to use attention or not
    def __init__(self, input_channels, use_dropout=False, use_bn=True, use_att=True):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout
        self.use_att = use_att
        if use_att:
            self.cbam = CBAM(gate_channels = input_channels//2)
    # parameters:
    # x: torch.tensor, input tensor of shape (batch_size, in_chan, height, width)
    # res: torch,tensor, tensor from the corresponding contract layer of shape 
    # (batch_size, in_chann//2, height', width'), which is likely needed to crop
    # so that can be concatenated to x
    def forward(self, x, skip_con_x):
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        if self.use_att:
            x = self.cbam(x)
        return x

# feature map: maps the output from the last expand block
# to a tensor of desired channels, in our case, 2 or 3 channels
# depends on whether we use rgb or lab to represent our image
# or maps the inputs from the begining to some desired channels
class FeatureMapBlock(nn.Module):
    # parameters:
    # in_chan: int, the number of channels of the inputs
    # out_chan: int, the number of channels of the desired outputs
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
    # parameters:
    # x: torch.tensor, of shape (batch_size, in_chan, height, weight)   
        x = self.conv(x)
        return x
    
    
# putting all blocks together to get our unet
# we will use the UNet architecture to colorize our image
# here the UNet_Classfier will not only colorize, but also will do
# classfications to feed some feature_vector into the UNet
class UNet_Classfier(nn.Module):
    # parameters:
    # in_chan: int, the number of channels of the inputs
    # out_chan: int, the number of channels of the desired outputs
    # output_dim: int, the dim of the classfier output(since the output will be a vector)(number of classes)
    # hidden_chan: int, the number of channels of tensors in the hidden layers
    # use_dp: bool, whether to use dropout or not
    # use_bn: bool, whether to use batch normalization or not
    # color_range, int(can only be 255 or 120)
    def __init__(self, in_chan, out_chan, output_dim, hidden_chan=32, color_range = 120):
        super(UNet_Classfier, self).__init__()
        self.color_range = color_range
        
        self.upfeature = FeatureMapBlock(in_chan, hidden_chan)
        self.contract1 = ContractingBlock(hidden_chan, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_chan * 2, use_dropout=True)
        self.contract3 = ContractingBlock(hidden_chan * 4, use_dropout=True)
        self.contract4 = ContractingBlock(hidden_chan * 8)
        self.contract5 = ContractingBlock(hidden_chan * 16)
        self.contract6 = ContractingBlock(hidden_chan * 32)
        self.contract6 = ContractingBlock(hidden_chan * 32)
        
        self.feature1 = FeatureMapBlock(hidden_chan*64, hidden_chan*32)
        self.feature2 = FeatureMapBlock(hidden_chan*32, hidden_chan*16)
        self.linear1 = nn.Linear(hidden_chan*32, hidden_chan*32)
        self.linear2 = nn.Linear(hidden_chan*32, output_dim)
        self.bn1 = nn.BatchNorm2d(hidden_chan * 32)
        self.bn2 = nn.BatchNorm2d(hidden_chan * 16)
        self.maxpool = nn.MaxPool2d(kernel_size = (2,2))
        self.feature_map = FeatureMapBlock(64*hidden_chan, 48*hidden_chan)
        
        self.expand0 = ExpandingBlock(hidden_chan * 64)
        self.expand1 = ExpandingBlock(hidden_chan * 32)
        self.expand2 = ExpandingBlock(hidden_chan * 16)
        self.expand3 = ExpandingBlock(hidden_chan * 8)
        self.expand4 = ExpandingBlock(hidden_chan * 4)
        self.expand5 = ExpandingBlock(hidden_chan * 2)
        self.downfeature = FeatureMapBlock(hidden_chan, out_chan)
        self.activation = nn.Tanh() if self.color_range == 120 else torch.sigmoid()
        
    # parameters:
    # x: torch.tensor, of shape (batch_size, in_chan, height, weight)
    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5) #x6.shape = (batch_size, 2048, 2, 2)
        
        classify = self.feature1(x6) #x.shape = (batch_size, 1024, 2, 2)
        nn.LeakyReLU(negative_slope=0.2, inplace=True)(classify)
        classify = self.bn1(classify)
        
        feature = self.feature2(classify) #x2.shape = (batch_size, 512, 2, 2)
        nn.LeakyReLU(negative_slope=0.2, inplace=True)(feature)
        feature = self.bn2(feature)
        
        output = self.maxpool(classify).squeeze(2).squeeze(2)
        output = self.linear1(output)
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
        output = self.linear2(output)
        
        x6_prime = self.feature_map(x6)
        x6_prime = torch.cat([x6_prime, feature], axis = 1)
        x7 = self.expand0(x6_prime, x5)
        x8 = self.expand1(x7, x4)
        x9 = self.expand2(x8, x3)
        x10 = self.expand3(x9, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)
        return output, self.activation(xn)*self.color_range # since 'lab' mode ranges from -120 to 120
    
# the discriminator in PatchGan, which utlize 
# the contract block in UNet, and will output
# a classifying matrix of shape (3(2),n,n) which, i,j-th
# element will corresponding to a patch of the image.
# In the original version of patchGAN, the matrix has shape
# (n,n,1), here, I want to experiment on using a three channels 
# tensor to let the discriminator gives more feedback on each 
# color channels.
class Discriminator(nn.Module):
    # parameters:
    # in_chan: int, the number of channels of the inputs
    # out_chan: int, the number of channels of the desired outputs
    # hidden_chan: int, the number of channels of tensors in the hidden layers   
    def __init__(self, input_channels, output_channels, hidden_channels=8):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.final = FeatureMapBlock(hidden_channels*16, output_channels)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn
    
