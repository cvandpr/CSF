import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import to_2tuple


# Borrowed from ''Improving image restoration by revisiting global information aggregation''
# --------------------------------------------------------------------------------
train_size = (1, 3, 256, 256)
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5,4,3,2,1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )
           
    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2]*self.base_size[0]//train_size[-2]
            self.kernel_size[1] = x.shape[3]*self.base_size[1]//train_size[-1]
            
            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0]*x.shape[2]//train_size[-2])
            self.max_r2 = max(1, self.rs[0]*x.shape[3]//train_size[-1])

        if self.fast_imp:   # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0]>=h and self.kernel_size[1]>=w:
                out = F.adaptive_avg_pool2d(x,1)
            else:
                r1 = [r for r in self.rs if h%r==0][0]
                r2 = [r for r in self.rs if w%r==0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:,:,::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h-1, self.kernel_size[0]//r1), min(w-1, self.kernel_size[1]//r2)
                out = (s[:,:,:-k1,:-k2]-s[:,:,:-k1,k2:]-s[:,:,k1:,:-k2]+s[:,:,k1:,k2:])/(k1*k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1,r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1,0,1,0)) # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:,:,:-k1,:-k2],s[:,:,:-k1,k2:], s[:,:,k1:,:-k2], s[:,:,k1:,k2:]
            out = s4+s1-s2-s3
            out = out / (k1*k2)
    
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w)//2, (w - _w + 1)//2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        
        return out
# --------------------------------------------------------------------------------
class BasicConv1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, gelu=False, bn=False, bias=True):
        super(BasicConv1, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.gelu = nn.GELU() if gelu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.gelu is not None:
            x = self.gelu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    
class SpatialGate(nn.Module):
    def __init__(self, channel):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv1(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, gelu=False)
        self.dw1 = nn.Sequential(
            BasicConv1(channel, channel, 5, stride=1, dilation=2, padding=4, groups=channel),
            BasicConv1(channel, channel, 7, stride=1, dilation=3, padding=9, groups=channel)
        )
        self.dw2 = BasicConv1(channel, channel, kernel_size, stride=1, padding=1, groups=channel)

    def forward(self, x):
        out = self.compress(x)
        out = self.spatial(out)
        out = self.dw1(x) * out + self.dw2(x)  # SSM
        return out


class LocalAttention(nn.Module):
    def __init__(self, channel, p) -> None:
        super().__init__()
        self.channel = channel

        self.num_patch = 2 ** p
        self.sig = nn.Sigmoid()

        self.a = nn.Parameter(torch.zeros(channel,1,1))
        self.b = nn.Parameter(torch.ones(channel,1,1))

    def forward(self, x):
        out = x - torch.mean(x, dim=(2,3), keepdim=True)
        return self.a*out*x + self.b*x

class ParamidAttention(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        pyramid = 1
        self.spatial_gate = SpatialGate(channel)
        layers = [LocalAttention(channel, p=i) for i in range(pyramid-1,-1,-1)]
        self.local_attention = nn.Sequential(*layers)
        self.a = nn.Parameter(torch.zeros(channel,1,1))
        self.b = nn.Parameter(torch.ones(channel,1,1))
    def forward(self, x):
        out = self.spatial_gate(x)
        out = self.local_attention(out)
        return self.a*out + self.b*x

##############


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



class Gap(nn.Module):
    def __init__(self, in_channel, mode) -> None:
        super().__init__()

        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        if mode[0] == 'train':
            self.gap = nn.AdaptiveAvgPool2d((1,1))
        elif mode[0] == 'test':
            if mode[1] == 'Indoor':
                self.gap = AvgPool2d(base_size=246)
            elif mode[1] == 'Outdoor':
                self.gap = AvgPool2d(base_size=210)

    def forward(self, x):
        x_d = self.gap(x)
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        x_d = x_d  * self.fscale_d[None, :, None, None]
        return x_d + x_h

# class SimpleGate(nn.Module):
#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=1)
#         return x1 * x2


###################

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mode, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter

        self.dyna = dynamic_filter(in_channel//2, mode) if filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel//2, mode, kernel_size=5) if filter else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)

        if self.filter:
            k3, k5 = torch.chunk(out, 2, dim=1)
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            out = torch.cat((out_k3, out_k5), dim=1)

        out = self.conv2(out)
        return out + x


class DFFM(nn.Module):
    def __init__(self, dim, division_ratio=4):
        super(DFFM, self).__init__()
        self.dim = dim
        self.dim_partial = int(dim // division_ratio)
        hidden_features = int(dim * 2)

        # self.dwconv_1 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(1),
        #                                   groups=self.dim_partial)
        # self.dwconv_3 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(3), padding=3,
        #                                   dilation=3, groups=self.dim_partial)

        # self.dwconv_5 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(5), padding=6,
        #                                   dilation=3, groups=self.dim_partial)
        # self.dwconv_7 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(7), padding=9,
        #                                   dilation=3, groups=self.dim_partial)

        self.dwconv_1 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(1),
                                  groups=self.dim_partial)
        self.dwconv_3 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(3), padding=1,
                                  dilation=1, groups=self.dim_partial)

        self.dwconv_5 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(5), padding=2,
                                  dilation=1, groups=self.dim_partial)
        self.dwconv_7 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(7), padding=3,
                                  dilation=1, groups=self.dim_partial)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_features, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_features, dim, 1, bias=False)
        )

        layer_scale_init_value = 0.
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.Conv1 = nn.Sequential(
            nn.Conv2d(dim, 2*dim,kernel_size=3,padding=1,stride=1,groups=dim),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(2*dim, dim,kernel_size=3,padding=1,stride=1,groups=dim))
        self.Conv1_1 = nn.Sequential(
            nn.Conv2d(dim, 2*dim, kernel_size=3, padding=1, stride=1, groups=dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2*dim, dim, kernel_size=3, padding=1, stride=1, groups=dim))
        self.Conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        # self.scale = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)


    def forward(self, x):
        input = x
        x_1, x_2, x_3, x_4 = torch.split(x, [self.dim_partial, self.dim_partial, self.dim_partial, self.dim_partial], dim=1)
        x_1 = self.dwconv_1(x_1)
        x_2 = self.dwconv_3(x_2)
        x_3 = self.dwconv_5(x_3)
        x_4 = self.dwconv_7(x_4)

        x = torch.cat((x_1, x_2, x_3, x_4), 1)
        x = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x) + input

        b, c, h, w = input.shape
        # a = 0.2
        a = 0.9
        mix = input + 1e-8
        mix_mag = torch.abs(mix)
        mix_pha = torch.angle(mix)
        # Ghost Expand
        mix_mag = self.Conv1(mix_mag)
        mix_pha = self.Conv1_1(mix_pha)

        re_main = mix_mag * torch.cos(mix_pha)
        img_main = mix_mag * torch.sin(mix_pha)
        x_out_main = torch.complex(re_main, img_main)
        x_out_main = torch.abs(torch.fft.irfft2(x_out_main, s=(h, w), norm='backward')) + 1e-8

        return self.Conv2(a * x_out_main + (1 - a) * x)
        # mix1 = self.mix1(x_out_main, x)
        # return self.Conv2(mix1)

###############
class dynamic_filter(nn.Module):
    def __init__(self, inchannels, mode, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.modulate = SFconv(inchannels, mode)

    def forward(self, x):
        identity_input = x 
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)
    
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        out = self.modulate(low_part, out_high)
        return out


"""Modulator"""
class SFconv(nn.Module):
    def __init__(self, features, mode, M=2, r=2, L=32) -> None:
        super().__init__()
        
        d = max(int(features/r), L)
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(features, features, 1, 1, 0)

        if mode[0] == 'train':
            self.gap = nn.AdaptiveAvgPool2d(1)
        elif mode[0] == 'test':
            if mode[1] == 'Indoor':
                self.gap = AvgPool2d(base_size=246)
            elif mode[1] == 'Outdoor':
                self.gap = AvgPool2d(base_size=210)

    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)
        
        attention_vectors = torch.cat([high_att, low_att], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)

        fea_high = high * high_att
        fea_low = low * low_att
        
        out = self.out(fea_high + fea_low) 
        return out



class Patch_ap(nn.Module):
    def __init__(self, mode, inchannel, patch_size):
        super(Patch_ap, self).__init__()

        if mode[0] == 'train':
            self.ap = nn.AdaptiveAvgPool2d((1,1))
        elif mode[0] == 'test':
            if mode[1] == 'Indoor':
                self.ap = AvgPool2d(base_size=246)
            elif mode[1] == 'Outdoor':
                self.ap = AvgPool2d(base_size=210)


        self.patch_size = patch_size
        self.channel = inchannel * patch_size**2
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.zeros(self.channel))

    def forward(self, x):

        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)

        low = self.ap(patch_x)
        high = (patch_x - low) * self.h[None, :, None, None]
        out = high + low * self.l[None, :, None, None]
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)

        return out

###############
class CGM(nn.Module):
    def __init__(self, in_channels, channels):
        super(CGM, self).__init__()
        self.conv_first_r = nn.Conv2d(in_channels // 3, channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_first_g = nn.Conv2d(in_channels // 3, channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_first_b = nn.Conv2d(in_channels // 3, channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.instance_r = nn.InstanceNorm2d(channels // 4, affine=True)
        self.instance_g = nn.InstanceNorm2d(channels // 4, affine=True)
        self.instance_b = nn.InstanceNorm2d(channels // 4, affine=True)

        self.act = nn.GELU()

        self.conv_out_r = nn.Conv2d(channels // 4, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_g = nn.Conv2d(channels // 4, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_b = nn.Conv2d(channels // 4, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.cov_out = nn.Conv2d(channels * 3, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x1, x2, x3 = torch.chunk(x, 3, dim=1)

        x_1 = self.conv_first_r(x1)
        x_2 = self.conv_first_g(x2)
        x_3 = self.conv_first_b(x3)

        out_instance_r = self.conv_out_r(self.act(self.instance_r(x_1)))
        out_instance_g = self.conv_out_g(self.act(self.instance_g(x_2)))
        out_instance_b = self.conv_out_b(self.act(self.instance_b(x_3)))

        out_instance = torch.cat((out_instance_r, out_instance_g, out_instance_b), dim=1)

        out = self.cov_out(out_instance)

        return out
