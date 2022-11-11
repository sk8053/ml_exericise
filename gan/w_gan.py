# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:18:06 2022

@author:seongjoon kang
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d,n_cond = 20):
        super(Discriminator, self).__init__()
        self.disc1 = nn.Sequential(
            # Input: N x channels_img x 64 x 50
            nn.Conv2d(
                channels_img+1, features_d, kernel_size=4, stride =2, padding=1
                ),                                         # 24 x 25
            nn.LeakyReLU(0.2),
            self._block(features_d, 2*features_d, 4, 2, 1), # 112 x 12
            self._block(features_d*2, 4*features_d, 4, 2, 1), # 6 x 6
            )
        self.disc2 = nn.Sequential(
            self._block(features_d*4, 8*features_d, 4, 2, 1), # 3 x 3
            nn.Conv2d(features_d*8,1,kernel_size= 4, stride =2, padding = 1), # 1 x1
           # nn.Sigmoid(),
            )
        self.fc = nn.Sequential(
            nn.Linear(n_cond, 48*50),
        )
        

        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias= False,
                ),
           # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine = True), # LayerNorm <-> Instance norm
            nn.LeakyReLU(0.2)
            )
    def forward(self, x,cond):
        cond_ = self.fc(cond)
   
        cond_ = cond_.reshape(-1,1,48,50)
        x = torch.concat([x,cond_],dim=1)
        y = self.disc1(x)
        
        #cond2 = self.fc2(cond)
        #cond2 = cond2.reshape(-1,1,6,6)
        #z = torch.concat([y,cond2],dim=1)
        z = y
        #cond = self.fc(cond)
        #cond = cond.reshape(-1,1,6,6)
        #y = torch.concat([y,cond],dim=1)
       
        return self.disc2(z)
    
class Generator(nn.Module):
    def __init__(self, z_dim,  channels_img, feature_g, n_cond = 20):
        super(Generator, self).__init__()
        self.gen1 = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim+50, feature_g*16, 3,1,0), #N*f_g * 3 * 3
            self._block(feature_g*16,feature_g*8, 4, 2, 1), #  6 *6
        )
        self.fc = nn.Sequential(
            nn.Linear(n_cond, 50),

            #nn.LeakyReLU(0.2),
            
            #nn.Linear(50, 36),
           #nn.ReLU()
        )
 
        
        self.gen2 = nn.Sequential(
            self._block(feature_g*8,feature_g*4, 4, 2, 1), # 12 * 12
            self._block(feature_g*4,feature_g*2, 4, 2, 1), # 24* 24
            nn.ConvTranspose2d(feature_g*2, channels_img, 
                               kernel_size=[4,6],
                               stride = 2,
                               padding =1),
            #nn.Tanh(), #[-1,1]
            nn.ReLU(),
            )
        
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, 
                               out_channels, 
                               kernel_size,
                               stride,
                               padding,
                               bias = False,),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
    def forward(self,x, cond):
        cond_ = self.fc(cond) # Batch_N * 20 * 1 * 1
        cond_ = cond_.reshape(-1,50,1,1)
        x = torch.concat([x,cond_],dim = 1)
        
        y = self.gen1(x)
       
        #cond2 = self.fc2(cond)
        #cond2 = cond2.reshape(-1,1,6,6)
        #y = torch.concat([y,cond2],dim=1)
        
        return self.gen2(y)
    
def initialize_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            
def test():
    N, in_channels, H, W = 8,3, 64, 64
    z_dim = 100
    x = torch.randn((N,in_channels,H,W ))
    disc = Discriminator(in_channels, 8)
    initialize_weight(disc)
    assert disc(x).shape == (N,1,1,1)
    gen= Generator(z_dim, in_channels, 8)
    initialize_weight(gen)
    z = torch.randn(N,z_dim,1,1)
    assert gen(z).shape == (N, in_channels, H, W)
    print('Success ')

#test()
#torch.save(gen.state_dict(), 'save_model/cond_gen_second.pt')            