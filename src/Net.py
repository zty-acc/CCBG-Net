import torch
import torch.nn as nn

from .cnn import CNN
from .transformer import Transformer
from .decoder import Decoder
from .fcm import FCM         
from .dsc import IDSC


import torch.nn.functional as F

from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn


from collections import OrderedDict
import torchvision.ops as ops
class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.mid_channels = mid_channels
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:
            self.up = nn.Sequential(
                                    nn.Conv2d(mid_channels, in_channels, 
                                              kernel_size=1, bias=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:
            self.relu = nn.ReLU(inplace= False)
        
    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        
        
       
        x = (1-sim_map)*x + sim_map*y
        
        return x

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    assert num_channels % groups == 0, "通道数必须可以被组数整除"
    
   
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    
    x = x.view(batch_size, num_channels, height, width)
    return x

    

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)

class HHFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.srfconv =nn.Conv2d(dim, dim, kernel_size=3,padding=1)
        
        self.mrfconv = nn.Conv2d(dim, dim, kernel_size=5, stride=1,padding=2)
        
        self.lrfconv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
       

        self.compress = dim // 2
        self.mid = self.compress * 3
        
        self.mid_bottle = nn.Conv2d(dim, self.compress, 1)
        self.sr_bottle = nn.Conv2d(dim, self.compress, 1)
        self.mr_bottle = nn.Conv2d(dim, self.compress, 1)
        self.lr_bottle = nn.Conv2d(dim, self.compress, 1)
        self.se1 = SEBlock(self.mid, self.mid // 4)
        self.se2 = SEBlock(self.mid, self.mid // 4)
        self.se3 = SEBlock(self.mid, self.mid // 4)
        self.last_conv = nn.Conv2d(self.compress, dim, 1)

        
        
        self.pagfm1 = PagFM(dim//2 ,dim//4,after_relu=True, with_channel=True)
        self.pagfm2 = PagFM(dim//2 ,dim//4,after_relu=True, with_channel=True)
        self.pagfm3 = PagFM(dim//2 ,dim//4,after_relu=True, with_channel=True)
        self.pagfm4 = PagFM(dim//2 ,dim//4,after_relu=True, with_channel=True)
        self.pagfm5 = PagFM(dim//2 ,dim//4,after_relu=True, with_channel=True)
        self.pagfm6 = PagFM(dim//2 ,dim//4,after_relu=True, with_channel=True)
        

        self.offset_conv_3x3 = nn.Conv2d(dim, 18, kernel_size=3, padding=1)
        self.deform_conv_3x3 = ops.DeformConv2d(dim, dim, kernel_size=3, padding=1)
    def forward(self, x,mid):
        srf = F.relu(self.srfconv(x))
        mrf = F.relu(self.mrfconv(x))
       
        lrf = F.relu(self.lrfconv(x))
        mid_copy = self.mid_bottle(mid)
        
        srf_e = self.sr_bottle(srf)
        
        mrf_e = self.mr_bottle(mrf)
        lrf_e = self.lr_bottle(lrf)
        weight1 = self.se1(srf_e)
        weight2 = self.se2(mrf_e)
        weight3 = self.se3(lrf_e)
        srf_e = srf_e * weight1
        mrf_e = mrf_e * weight2
        lrf_e = lrf_e * weight3
        
        
        F_step1 =  self.pagfm1(mid_copy, srf_e)
        
        F_step2 =  self.pagfm2(F_step1, mrf_e)

       
        F_step3 =  self.pagfm3(F_step2, lrf_e)
        
        N_step1 = self.pagfm4(mid_copy, lrf_e)
        
        N_step2 = self.pagfm5(N_step1, mrf_e)
        
        N_step3 = self.pagfm6(N_step2, srf_e)
        
        feature = torch.concat([F_step3,N_step3],dim=1)
        
        output_tensor = channel_shuffle(feature, groups=4)
        offset_3x3 = self.offset_conv_3x3(output_tensor)
        attn1 = self.deform_conv_3x3(output_tensor,offset_3x3)
        
        finally_tensor = attn1 + mid
        
        return finally_tensor

####################################################
class UP(nn.Module):
    def __init__(self, in_channels):
        super(UP, self).__init__()
    
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=1)
        
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)      
        x = self.up(x)        
        return x
######################################################
class DownCo(nn.Module):
    def __init__(self, in_channels):
        super(DownCo, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
######################################################

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = CNN()
        self.encoder2 = Transformer()

        self.fuse1 = FCM(64)
        self.fuse2 = FCM(128)
        self.fuse3 = FCM(256)
        

        self.Conv = nn.Sequential(IDSC(1024, 512),
                                  nn.BatchNorm2d(512),
                                  nn.GELU())
        self.decoder = Decoder()

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(512, 512)
#####################################################
        self.up1 = UP(128)
        self.up2 = UP(256)
        self.down1 = DownCo(64)
        self.down2 = DownCo(128)
        self.fusion1 = HHFM(64)
        self.fusion2 = HHFM(128)
        self.fusion21 = HHFM(128)
        self.fusion3 = HHFM(256)
    def forward(self, x):
        x1, x2, x3, x4, out1 = self.encoder1(x)
        y1, y2, y3, out2 = self.encoder2(x)

        f1 = self.fuse1(x2, y1)
        
        f2 = self.fuse2(x3, y2)
        
        f22up = self.up1(f2)
        f11 = self.fusion1(f22up,f1)
        
        f3 = self.fuse3(x4, y3)
        
        f1down = self.down1(f1)
        f3up = self.up2(f3)
        f221 = self.fusion2(f3up,f2)
        f222 = self.fusion21(f1down,f2)
       
        f2222 = (f221 + f222+f2)
        
        f22down = self.down2(f2)
        f33 = self.fusion3(f22down,f3)
        
        B1, C1, H1, W1 = out1.shape
        B2, C2, H2, W2 = out2.shape
        x_temp = self.avg(out1)
        y_temp = self.avg(out2)
        x_weight = self.linear(x_temp.reshape(B1, 1, 1, C1))
        y_weight = self.linear(y_temp.reshape(B2, 1, 1, C2))
        x_temp = out1.permute(0, 2, 3, 1)
        y_temp = out2.permute(0, 2, 3, 1)
        x1 = x_temp * x_weight
        y1 = y_temp * y_weight

        x1 = x1.permute(0, 3, 1, 2)
        y1 = y1.permute(0, 3, 1, 2)


        out = torch.cat([x1, y1], dim=1)
        out = self.Conv(out)
        
        mask,boundary_get1,boundary_get2,boundary_get3,completebody1,completebody2,completebody3,tem10 = self.decoder(out, f11, f2222, f33)
        return mask,boundary_get1,boundary_get2,boundary_get3,completebody1,completebody2,completebody3,tem10

