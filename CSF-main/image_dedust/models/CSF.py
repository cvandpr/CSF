import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


# Encoder Block
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res, mode):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel, mode) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, mode, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Decoder Block
class DBlock(nn.Module):
    def __init__(self, channel, num_res, mode):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel, mode) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, mode, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    """ combine two features in designed dimension"""
class FFusion(nn.Module):
    def __init__(self, channel, height=2, reduction=8):
        super(FFusion, self).__init__()
        self.height = height
        d = max(int(channel / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, d, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d, channel * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class CSF(nn.Module):
    def __init__(self, mode, num_res=8):
        super(CSF, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, mode),
            EBlock(base_channel*2, num_res, mode),
            EBlock(base_channel*4, num_res, mode),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, mode),
            DBlock(base_channel * 2, num_res, mode),
            DBlock(base_channel, num_res, mode)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.ff1 = FFusion(base_channel * 4)
        self.ff2 = FFusion(base_channel * 2)

        self.cg1 = CGM(3, base_channel * 4)
        self.cg2 = CGM(3, base_channel * 2)

        self.bn = DFFM(base_channel * 4)

    def forward(self, x):
        # encoder
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.cg2(x_2)
        z4 = self.cg1(x_4)

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128*128
        z = self.feat_extract[1](res1)
        z = self.ff2([z, z2])
        res2 = self.Encoder[1](z)
        # 64*64
        z = self.feat_extract[2](res2)
        z = self.ff1([z, z4])
        z = self.Encoder[2](z)

        z = self.bn(z)          # bottleneck

        # decoder
        z = self.Decoder[0](z)
        # 128*128
        z = self.feat_extract[3](z)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)        # conv 1*1
        z = self.Decoder[1](z)
        # 256*256
        z = self.feat_extract[4](z)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)        # conv 1*1
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs = z+x

        return outputs


def build_net(mode):
    return CSF(mode)
