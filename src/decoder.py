import torch
import torch.nn as nn
import numbers
from einops import rearrange
from .transformer import Block

from .dsc import IDSC

import cv2 as cv
import numpy as np

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import ConvModule
# from mmseg.registry import MODELS
# from ..utils import resize

# 定义一个简单的卷积层
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    
    def forward(self, x):
        return self.conv(x)

# 定义整个网络模块
class CustomModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomModule, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        
        self.con1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.con2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, Xi, X_hat_next):
        
        H_tilde_i = torch.cat((Xi, X_hat_next), dim=1)  # 通道维度拼接concat
        
        pooled = self.avg_pool(H_tilde_i) # AvgPool
        
        w = Xi.shape[-1]
        h = Xi.shape[-2]
        c= Xi.shape[-3]
        conv_out = F.interpolate(pooled, (h, w), mode='bilinear')
        conv_out = self.con2(conv_out)
         
        sig_out = self.sigmoid(conv_out)
        
        H_i = H_tilde_i - H_tilde_i * sig_out
        
        H_i_boun = self.con1(H_i)
        return H_i_boun
#############################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape)).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        sigma = x.var(-1, keepdim=True, unbiased=False).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)    

class RefineAttention(nn.Module):
    
    def __init__(self,dim):
        super(RefineAttention, self).__init__()
        # self.num_heads = num_heads   
        self.num_heads=8
        #dim = 96
        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)   
    def forward(self, x, mask_d,mask_e):
        b,c,h,w = x.shape   
        x = self.norm(x)
        
        y1 = x*mask_d
        y2 = x*mask_e
        y1_sigmoid = torch.sigmoid(y1)
        y2_sigmoid = torch.sigmoid(y2)
        out_sa = x.clone()
        with torch.no_grad():
            for i in range(b):
                z_d = []
                z_e = []
                pos_d = np.argwhere(y1_sigmoid[i][0].cpu().detach().numpy() == 0.5)
                pos_e = np.argwhere(y2_sigmoid[i][0].cpu().detach().numpy() == 0.5)            
                for j in range(c):
                    z_d.append(y1[i,j,pos_d[:,0],pos_d[:,1]])
                    z_e.append(y2[i,j,pos_e[:,0],pos_e[:,1]])
                
                z_d = torch.stack(z_d)
                z_e = torch.stack(z_e)
                z_e = z_e.cuda()
                z_d = z_d.cuda()
                k1 = rearrange(z_e, '(head c) z -> head z c', head=self.num_heads)
                v1 = rearrange(z_e, '(head c) z -> head z c', head=self.num_heads)
                q1 = rearrange(z_d, '(head c) z -> head z c', head=self.num_heads)    
                q1 = torch.nn.functional.normalize(q1, dim=-1)
                k1 = torch.nn.functional.normalize(k1, dim=-1)   
                
                attn1 = (q1 @ k1.transpose(-2, -1))
                attn1 = attn1.softmax(dim=-1) 
                out1 = (attn1 @ v1) + q1  
                
                out1 = rearrange(out1, 'head z c -> (head c) z', head=self.num_heads)
                for j in range(c):
                    out_sa[i,j,pos_d[:,0],pos_d[:,1]] = out1[j]      
        # channel att
        k2 = rearrange(y2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(y2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = rearrange(y1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)    
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)      
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)   
        out2 = (attn2 @ v2) + q2      
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)   
        out = x + out_sa + out2
        return out
################################################### boundary 
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.t3 = IDSC(384, 256)
        self.t2 = IDSC(192, 128)
        self.t1 = IDSC(96, 64)

        self.block3 = Block(256, window_size=2, alpha=0.2)
        self.block2 = Block(128, window_size=2, alpha=0.3)
        self.block1 = Block(64, window_size=2, alpha=0.4)

        self.boundary_get3 = CustomModule(384,1)
        self.boundary_get2 = CustomModule(192,1)
        self.boundary_get1 = CustomModule(96,1)

        self.boundary_att3 = RefineAttention(256)
        self.boundary_att2 = RefineAttention(128)
        self.boundary_att1 = RefineAttention(64)
        self.up = nn.PixelShuffle(2)

        
        self.final = nn.Sequential(
                                nn.PixelShuffle(4),
                                IDSC(4, 1),
                               
                                )
        self.final3 = nn.Sequential(
                                   IDSC(64, 1),
                                   )
        self.final2 = nn.Sequential(
                                   IDSC(32, 1),
                                   )
        self.final1 = nn.Sequential(
                                   IDSC(64, 1),
                                   )
        self.final0 = nn.Sequential(
                                   IDSC(128, 1),
                                   )
    def forward(self, x, x1, x2, x3):
        temp = self.up(x) 
        
        templ0 =self.final0(temp) 
        
        boundary_get3 = self.boundary_get3(temp,x3) 
       
        boundary_att3 = self.boundary_att3(x3,boundary_get3,templ0)
        
        temp = torch.cat([boundary_att3, temp], dim=1) #2x384x32x32
       
        temp = self.t3(temp) 
        
        temp = self.up(temp) 
        
        completebody3  =self.final3(temp)
        boundary_get2 = self.boundary_get2(temp,x2) 
        
        boundary_att2 = self.boundary_att2(x2,boundary_get2,completebody3)
        
        temp = torch.cat([boundary_att2, temp], dim=1) 
        
        temp = self.t2(temp) 
        
        
        temp = self.up(temp) 
        
        completebody2  =self.final2(temp)
        boundary_get1 = self.boundary_get1(temp,x1) 
        boundary_att1 = self.boundary_att1(x1,boundary_get1,completebody2)
        temp = torch.cat([boundary_att1, temp], dim=1)# 
        completebody1  =self.final1(boundary_att1)
        
        temp = self.t1(temp) 

        out = self.final(temp) 
        return out,boundary_get1,boundary_get2,boundary_get3,completebody1,completebody2,completebody3,templ0