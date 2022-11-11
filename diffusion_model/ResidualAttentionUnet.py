# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:36:26 2022

@author: seongjoon kang
"""

import torch
from torch import nn
import math
import torchvision.transforms as T
from attention import AttentionBlock

class Block(nn.Module):
    
    def __init__(self, in_ch, out_ch, time_emb_dim,dist_2d_emb_dim, height_emb_dim, block_type = 'up'):
        super().__init__()
        self.time_fc =  nn.Linear(time_emb_dim, out_ch)
        self.dist_2d_fc = nn.Linear(dist_2d_emb_dim, out_ch)
        self.height_fc = nn.Linear(height_emb_dim, out_ch)
        #d_k = int(out_ch/8)
        # up blocks
        if block_type == 'up':
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1,
                                   bias = False)
            self.res_conv = nn.Conv2d(in_ch*2, out_ch, 1, padding=0, 
                                      bias = False)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1, bias = False)
            #self.att = AttentionBlock(out_ch,d_k = d_k, n_groups=4)
        # down blocks
        elif block_type == 'down' :
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1,bias = False)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1, bias = False)
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1, padding=0, 
                                      bias = False)
            #self.att = AttentionBlock(out_ch,d_k = d_k, n_groups=4)
            
        # middle blocks 
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1,bias = False)
            self.transform = nn.Conv2d(out_ch, in_ch, 4, 2, 1, bias = False)
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, 
                                      padding=1, bias = False)
            #self.att = AttentionBlock(in_ch, d_k = d_k, n_groups=4)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1,bias = False)
        # group normalization
        # https://arxiv.org/abs/1803.08494
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=out_ch) 
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=out_ch)
        self.relu  = nn.SiLU()
        self._block_type = block_type
       
    def forward(self, x, dist_2d, height,t):
       
        h_res = self.res_conv(x)
        
        # First Conv
        h = self.norm1(self.relu(self.conv1(x))) 
        # Time embedding
        time_emb = self.relu(self.time_fc(t))
        
        dist_2d_emb = self.relu(self.dist_2d_fc(dist_2d))
        height_emb = self.relu(self.height_fc(height))
        
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2] #increase dimesnion by 2
        dist_2d_emb = dist_2d_emb[(..., ) + (None, ) * 2]
        height_emb = height_emb[(..., ) + (None, ) * 2]
        #print(time_emb.size(), dist_2d_emb.size(), height_emb.size())
        # Add time channel
        h = h + time_emb + dist_2d_emb + height_emb
       
        # Second Conv
        h = self.norm2(self.relu(self.conv2(h)+h_res))
        
      
        # Down or Upsample      
        output = self.transform(h)
      
        #output = self.att(output)
       
        return output



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings

class Attention (nn.Module):
    
    # proposed attention structure in the below link
    # https://arxiv.org/abs/1804.03999
    def __init__(self,  g_in_ch, skip_in_ch):
        super().__init__()
        self.conv_gate = nn.Conv2d(g_in_ch, g_in_ch, kernel_size=(1,1), 
                                   bias = False)
        self.conv_skip_x  = nn.Conv2d(skip_in_ch, g_in_ch, stride=(2,2), 
                                      kernel_size=(2,2), padding =0, bias = False)
        self.relu = nn.SiLU()
        self.combine_conv = nn.Conv2d(g_in_ch, 1, kernel_size=(1,1), 
                                      bias = False)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.ConvTranspose2d(1, 1, 4,2,1, 
                                           bias = False)
        
    def forward(self, gate_x, skip_x):
        
        gate_x = self.conv_gate(gate_x)
        
        skip_x = self.conv_skip_x(skip_x)
        
        x = gate_x + skip_x
        x = self.relu(x)
        x = self.combine_conv(x)
        x = self.sigmoid(x)
        x = self.upsample(x)
        
        return x
        
class ResidualAttentionUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 1
        #down_channels = (64, 128, 256, 512, 1024)
        #up_channels = (1024, 512, 256, 128, 64)
        #down_channels = (16, 32, 64,128, 256)
        #up_channels = (256, 128, 64,32,  16)
        down_channels = (8,16, 32, 64,128)
        up_channels = (128, 64,32,  16,8)
        
        
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_fc = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU()
            )
        dist_2d_emb_dim = 22
        self.dist_2d_fc = nn.Sequential(
            SinusoidalPositionEmbeddings(dist_2d_emb_dim),
            nn.Linear(dist_2d_emb_dim, dist_2d_emb_dim),
            nn.SiLU()
            )
        
        height_emb_dim = 10
        self.height_fc = nn.Sequential(
            SinusoidalPositionEmbeddings(height_emb_dim),
            nn.Linear(height_emb_dim, height_emb_dim),
            nn.SiLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1,bias = False)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim, dist_2d_emb_dim, height_emb_dim,  block_type = 'down') \
                    for i in range(len(down_channels)-1)])
        
        # Upsample
        Block_list = []
        Attention_list = []
        for i in range(len(up_channels)-1):         
        
            Block_list.append(Block(up_channels[i], up_channels[i+1], \
                                    time_emb_dim, dist_2d_emb_dim, height_emb_dim, block_type = 'up'))
         
            if i != len(up_channels)-1-1:
                att = Attention(up_channels[i],up_channels[i+1])
                Attention_list.append(att)
                
       
        
        self.ups = nn.ModuleList(Block_list)
        
        # Middle block
        self.middle = nn.ModuleList([Block(down_channels[-1], down_channels[-1]*2, \
                                    time_emb_dim,  dist_2d_emb_dim, height_emb_dim, block_type = 'middle')])
            
        self.Attention_list = nn.ModuleList(Attention_list)
        self.output = nn.Conv2d(up_channels[-1], image_channels, out_dim)
        
        
    def forward(self, x, timestep, dist_2d, height):
        # Embedd time
        t = self.time_fc(timestep)   
        
        dist_2d = self.dist_2d_fc(dist_2d)
        height = self.height_fc(height)
        
        # Initial conv     
        x = self.conv0(x)
       
        # Unet
        skip_connections = []
        # Down blocks
        for down in self.downs:
            x = down(x, dist_2d, height,t)
            skip_connections.append(x)
        gate_x = None
       
        # Middle block
        for middle in self.middle:
            x = middle(x, dist_2d, height,t)
        
        # Up blocks
        for i, up in enumerate(self.ups):
            # take skip connections
            skip_x = skip_connections.pop()
            
            # apply attention here 
            # between skip connection and gated output
            # https://arxiv.org/abs/1804.03999
            if i>0:
                attention = self.Attention_list[i-1] 
                w = attention( gate_x, skip_x)
                if w.shape != skip_x.shape:
                    w = T.Resize(size=skip_x.shape[2:])(w)
                skip_x = skip_x * w
            
            if x.shape != skip_x.shape:
                x = T.Resize(size = skip_x.shape[2:])(x)
            
            gate_x = x    
            
            x = torch.cat((x, skip_x), dim=1)
            x = up(x, dist_2d, height, t)
            
        return self.output(x)