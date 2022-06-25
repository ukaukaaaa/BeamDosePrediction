import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x



class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x



class Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(Net, self).__init__()
        self.has_dropout = has_dropout

        self.block_one      = ConvBlock(1, n_channels, 16, normalization=normalization)
        self.block_one_dw   = DownsamplingConvBlock(16, 32, normalization=normalization)

        self.block_two      = ConvBlock(2, 32, 32, normalization=normalization)
        self.block_two_dw   = DownsamplingConvBlock(32, 64, normalization=normalization)

        self.block_three    = ConvBlock(3, 64, 64, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(64, 128, normalization=normalization)

        self.block_four     = ConvBlock(3, 128, 128, normalization=normalization)
        self.block_four_dw  = DownsamplingConvBlock(128, 256, normalization=normalization)

        self.block_five     = ConvBlock(3, 256, 256, normalization=normalization)
        self.block_five_up  = UpsamplingDeconvBlock(256, 128, normalization=normalization)

        self.block_six      = ConvBlock(3, 128+128, 128, normalization=normalization)
        self.block_six_up   = UpsamplingDeconvBlock(128, 64, normalization=normalization)

        self.block_seven    = ConvBlock(3, 64+64, 64, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(64, 32, normalization=normalization)

        self.block_eight    = ConvBlock(2, 32+32, 32, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(32, 16, normalization=normalization)

        self.block_nine     = ConvBlock(1, 16+16, 16, normalization=normalization)
        
        self.out_conv       = nn.Conv3d(16, n_classes, 1, padding=0)
                
        self.activation = nn.Sigmoid()

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up   = self.block_five_up(x5)
        x5_up   = torch.cat((x5_up, x4), dim=1)
        x6      = self.block_six(x5_up)

        x6_up   = self.block_six_up(x6)
        x6_up   =  torch.cat((x6_up, x3), dim=1)
        x7      = self.block_seven(x6_up)

        x7_up   = self.block_seven_up(x7)
        x7_up   =  torch.cat((x7_up, x2), dim=1)
        x8      = self.block_eight(x7_up)

        x8_up   = self.block_eight_up(x8)
        x8_up   =  torch.cat((x8_up, x1), dim=1)
        x9      = self.block_nine(x8_up)

        if self.has_dropout:
            x9  = self.dropout(x9)

        x10     = self.out_conv(x9)
        out     = self.activation(x10)


        return out


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        # pdb.set_trace()
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out

        
    def model_name(self):
        return 'Network'

    def model_description(self):
        return 'Network for dose prediction. Global Dose Network and Beam-wise Dose Network has same structure but different output channels'
