# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:42:21 2022

@author: seongjoon kang
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from beta_scheduler import linear_beta_schedule
import numpy as np
from utill import show_tensor_image

class train_model():
    # train model by forward process 
    # and sample data by reverse process from model
    
    def __init__(self, model = None, total_timesteps:int =150, beta_max:float = 0.08, device = 'cpu'):
        '''
        Parameters
        ----------
        model : ResidualAttentionUnet
            Unet to denoise image across all the time steps
        total_timesteps : int, 
            total diffusion steps
        beta_max : float, optional
            maximum variance used for variance schedule
        device : TYPE, optional
            device to use,cuda or cpu 
            The default is 'cpu'.

        '''
        self.model = model
        self.model.train()
        self.total_timesteps = total_timesteps
        self.betas = linear_beta_schedule(timesteps=total_timesteps, end = beta_max)
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_bar_t = torch.cumprod(self.alphas, axis=0) # alpha_bar_t in the paper
        self.alphas_bar_t_1 = F.pad(self.alphas_bar_t[:-1], (1, 0), value=1.0) # alpha_bar_t-1 in the paper

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar_t) # mean of q(x_t|x_0) in Eq.(4)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar_t) # std of q(x_t|x_0) in Eq.(4)
        # posterior variance
        self.sigmas = self.betas * (1. - self.alphas_bar_t_1) / (1. - self.alphas_bar_t) # beta_telda , Eq. (7) 
        self.device = device
    
    @torch.no_grad()
    def forward_data(self,x_0, t):
        '''
        forward data and get noisy image using closed form from original image x_0
        
        Parameters
        ----------
        x_0 : torch tensor [batch size x channels x height x width]
            image data
        t : pytorch tensor array [batch_size]
            time steps where corresponding images are forwarded

        Returns
        -------
        x_t :torch tensor [batch size x channels x height x width]
            output of diffusion steps
        noise : pytorch tensor [batch size x channels x height x width]
            noise used for diffusion at each time steps
        '''
        device = self.device
        x_0 = x_0.to(device)
        noise = torch.randn_like(x_0).to(device)
        #sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_alphas_bar_t = self.sqrt_alphas_bar[t][:,None][:,None][:,None].to(device)
        
        
        sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[t][:,None][:,None][:,None].to(device)
        # compute noised image according to Eq. (4)
        x_t = sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t *noise
        # mean + variance
        return x_t, noise    

        
    def get_loss(self,model, x_0, cond, t):
        '''
        

        Parameters
        ----------
        model : Unet model
            
        x_0 : torch tensor [batch size x channels x height x width]
            image data
        cond : torch tensor [bath size, 2]
            conditional variable, 2-d distacne and height
        t : torch tensor array [batch_size]
            time steps where corresponding images are forwarded
  

        Returns
        -------
        loss : torch float
            L1 or L2 loss 
        loss_list : torch tensor array [batch_size]
            loss values corresponding to the time steps 

        '''
        x_noisy, noise = self.forward_data(x_0, t)
        x_noisy = torch.reshape(x_noisy,(-1,1,x_noisy.shape[2],x_noisy.shape[3]))
        
        noise_pred = model(x_noisy, cond[:,0], cond[:,1], t)
        
        loss = F.l1_loss(noise, noise_pred, reduction = 'sum')/x_0.shape[0]
        #loss = F.huber_loss(noise, noise_pred, reduction = 'sum')/x_0.shape[0]
        #loss = F.smooth_l1_loss(noise, noise_pred, reduction = 'sum')/x_0.shape[0]
        loss_list = torch.norm( noise - noise_pred,p=1, dim=(-2,-1))
        loss_list = torch.squeeze(loss_list)
          
        return loss, loss_list

    @torch.no_grad()
    def sample_data(self,x, cond, t):
        '''
        sampling data by reverse processs
        get x_t from x_t-1
        Parameters
        ----------
        x_0 : torch tensor [batch size x channels x height x width]
            image data
        t : torch tensor array [batch_size]
            time steps where corresponding images are reversed

        Returns
        -------
        TYPE
            x_t: denoised data at time t from t-1

        '''

        device = self.device
        betas_t = self.betas[t][:,None][:,None].to(device)

        # sample sqrt_one_minus_alphas corresponding to time steps
        sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas_bar[t][:,None][:,None].to(device) 
        # sample the reverse of alphas corresponding to time steps
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:,None][:,None].to(device)
        # sample the sigams corresponding to time steps (posterior variance)
        sigma_t = self.sigmas[t][:,None][:,None].to(device)

        # reparamaterize the output of model accorting to equation (11)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, cond[:,0], cond[:,1], t) / sqrt_one_minus_alphas_t
        )
        
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x).to(device)
            return model_mean + torch.sqrt(sigma_t) * noise

    @torch.no_grad()
    def sample_plot_image(self,img_size:tuple(), cond):
        # Sample noise
        
        img = torch.randn((1, 1, img_size[0], img_size[1]), device=self.device)
        plt.figure(figsize=(15,15))
        plt.axis('off')
        num_images = 10
        stepsize = int(self.total_timesteps/num_images)
        self.model.eval()
        for i in range(0,self.total_timesteps)[::-1]:
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            img = self.sample_data(img, cond, t)
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i/stepsize)+1)
                show_tensor_image(img.detach().cpu(), i)
                
        plt.colorbar()
        plt.show()
        
    @torch.no_grad()
    def sample_dataset(self, n_img:int=10, total_timestep:int = 150, cond = None,   img_size:tuple() = None):
        # Sample noise
        img = torch.randn((n_img, 1, img_size[0], img_size[1]), device=self.device)
        
        for i in range(0,total_timestep)[::-1]:
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            img = self.sample_data(img, cond, t)
        
        #path_loss = torch.squeeze(img[:,:,41,:])[:,:25]
        pl = img[:,:,np.arange(8),:].cpu()
        #p_1 = pl[:,:,:, np.arange(50, step =2)]
        p_2 = pl[:,:,:, np.arange(50, step =2)+1]
        #p_3 = torch.concat([p_1,p_2],dim = -2)
        path_loss = torch.mean(p_2, dim = -2)
        
        return path_loss
