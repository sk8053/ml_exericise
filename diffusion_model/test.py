# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:21:30 2022

@author: seongjoon kang
"""
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from ResidualAttentionUnet import ResidualAttentionUnet
from beta_scheduler import cosine_beta_schedule, sigmoid_beta_schedule, linear_beta_schedule, biquadratic_beta_schedule

device = "cuda" if torch.cuda.is_available() else "cpu"
total_timesteps = 150
beta_max = 0.08
betas = linear_beta_schedule(timesteps=total_timesteps, end = beta_max)
#betas = biquadratic_beta_schedule(total_time_steps, end = beta_max)

model_name = 'diffusion_model.pt'
#model_name = 'diffusion_model_hpc.pt'


# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_bar_t = torch.cumprod(alphas, axis=0) # alpha_bar_t in the paper
alphas_bar_t_1 = F.pad(alphas_bar_t[:-1], (1, 0), value=1.0) # alpha_bar_t-1 in the paper
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_bar = torch.sqrt(alphas_bar_t) # mean of q(x_t|x_0) in Eq.(4)
sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar_t) # std of q(x_t|x_0) in Eq.(4)
# posterior variance
sigmas = betas * (1. - alphas_bar_t_1) / (1. - alphas_bar_t) # beta_telda , Eq. (7) 

model = ResidualAttentionUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
model.load_state_dict(torch.load('save_model/%s'%model_name)['model'])
model.to(device)

@torch.no_grad()
def sample_data(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    
    #device = t.device()
    betas_t = betas[t][:,None][:,None].to(device)

    # sample sqrt_one_minus_alphas corresponding to time steps
    sqrt_one_minus_alphas_t = sqrt_one_minus_alphas_bar[t][:,None][:,None].to(device) 
    # sample the reverse of alphas corresponding to time steps
    sqrt_recip_alphas_t = sqrt_recip_alphas[t][:,None][:,None].to(device)
    # sample the sigams corresponding to time steps
    sigma_t = sigmas[t][:,None][:,None].to(device)

    # reparamaterize the output of model accorting to equation (11)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_t
    )
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x).to(device)
        return model_mean + torch.sqrt(sigma_t) * noise 

@torch.no_grad()
def sample_dataset(n_img:int=10, total_timestep:int = 150,  img_size:Tuple = None):
    # Sample noise
    img = torch.randn((n_img, 1, img_size[0], img_size[1]), device=device)
    
    for i in range(0,total_timestep)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_data(img, t)
    
    #path_loss = torch.squeeze(img[:,:,41,:])[:,:25]
    pl = img[:,:,np.arange(8)+39,:].cpu()
    p_1 = pl[:,:,:, np.arange(50, step =2)]
    p_2 = pl[:,:,:, np.arange(50, step =2)+1]
    p_3 = torch.concat([p_2],dim = -2)
    path_loss = torch.mean(p_3, dim = -2)
    
    return img, path_loss

with open('data_all_1.pickle','rb') as f:
    dataset_1 = pickle.load(f)

with open('data_all_2.pickle','rb') as f:
    dataset_2 = pickle.load(f)


dataset = np.append(dataset_1, dataset_2, axis =0)
pl = dataset[:,:,0,:]
pl = (pl[:,:,:25]+1)*77
pl = pl.reshape(-1)
#pl = pl.detach().cpu().numpy()

img, pl_model0 = sample_dataset(n_img = 1000, total_timestep = total_timesteps, img_size = dataset.shape[2:])
#pl_model = pl_model[:,:,:25]*160
pl_model = (pl_model0.reshape(-1)+1)*77
pl_model = pl_model.detach().cpu().numpy()

plt.figure()

plt.plot(np.sort(pl), np.linspace(0,1,len(pl)), label = 'data')
plt.plot(np.sort(pl_model), np.linspace(0,1, len(pl_model)), label = 'model')
plt.grid()
plt.legend()
plt.show()
