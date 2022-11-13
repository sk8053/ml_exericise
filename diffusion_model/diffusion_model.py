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
from scipy.stats import norm

class diffusion_model():
    # train model by forward process 
    # and sample data by reverse process from model
    
        ## all computations are based on the following papers.
        ## https://arxiv.org/abs/2102.09672
        ## https://arxiv.org/abs/2010.02502
    
    def __init__(self, model = None, total_timesteps:int =150, beta_max:float = 0.08, device = 'cpu',
                 beta_schedule = linear_beta_schedule):
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
        beta_schedule:
            beta schedule types, default is linear schedule

        '''
        self.model = model
        self.model.train()
        self.total_timesteps = total_timesteps
        #self.betas = linear_beta_schedule(timesteps=total_timesteps, end = beta_max).to(device)
        self.betas = beta_schedule(timesteps=total_timesteps, end = beta_max).to(device)
        self.betas = self.betas[(..., ) + (None, ) * 3]
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        # alpha_bar_t in the paper
        self.alphas_bar_t = torch.cumprod(self.alphas, axis=0) 
        # alpha_bar_t-1 in the paper
        self.alphas_bar_t_1 = F.pad(self.alphas_bar_t[:-1], (0, 0, 0, 0, 0,0 ,1,0), value=1.0) 
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # mean of q(x_t|x_0) in Eq.(4)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar_t)
        # variance of q(x_t|x_0) in Eq.(4)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar_t) 
        # posterior variance, denoted as beta_telda 
        self.sigmas = self.betas * (1. - self.alphas_bar_t_1) / (1. - self.alphas_bar_t) 
        self.device = device
        # gaussian function : N(0,1)
        self.gaussian_rv = norm()
        

     
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
        
        # sqrt(alpha_bar_t)
        sqrt_alphas_bar_t = self.sqrt_alphas_bar[t]
        # sqrt(1-alpha_bar_t)
        sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[t]
        # compute noised image according to Eq. (2)
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

        # all equation numbers come from https://arxiv.org/abs/2102.09672
        device =self.device
        x_t, noise = self.forward_data(x_0, t)
        x_t = torch.reshape(x_t,(-1,1,x_t.shape[2],x_t.shape[3]))
        
        # epsilon_theta from model 
        noise_pred = model(x_t, cond[:,0], cond[:,1], t)
        # model outputs variable v so that it can learn Sigma(x_t, t)
        # equation (15) in the paper
        noise_pred, v = torch.split(noise_pred, 1, dim = 1)
        
        # firsty compute L0 in the paper, -log(p(x0|x1))
        # we simply use gaussian pdf
        KL_list = torch.zeros(len(t)).to(device)
        KL_0 = 0
        if 0 in t:
            ind = torch.where(0==t)[0]
            e = noise[ind] - noise_pred[ind]
            prob = self.gaussian_rv.pdf(e.detach().cpu().numpy())
            prob = torch.tensor(prob, dtype = torch.float)
            prob = prob.clamp(min=1e-8, max=1e8)
            KL_0 = -1*torch.log(prob).sum()
            KL_list[ind] = KL_0
            
        # 1/sqrt(alpha_t)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        # betas used for diffusion process
        betas_t = self.betas[t]
        # alpha_bar_t -1
        alphas_bar_t_1 = self.alphas_bar_t_1[t]
        # alpha_bar_t
        alphas_bar_t = self.alphas_bar_t[t]
        alpha_t = self.alphas[t]
        # 1/sqrt(alpha_bar_t)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        # sqrt(1-alpha_bar_t)
        sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas_bar[t]
        
        # compute posterior 1) variance and 2) mean of q(x_t-1|x_t, x_0)
        # 1) posterior variance of of q(x_t-1|x_t, x_0) 
        # denoted as sigma_telda_t in equation (10)
        sigma_telda = self.sigmas[t]
        sigma_telda = sigma_telda.clamp(1e-6)
        sigma_telda_log = torch.log(sigma_telda)
        
        # 2) mean of q(x_t-1|x_t, x_0) from equation (11) 
        mean_telda = x_0*torch.square(alphas_bar_t_1)*betas_t/(1-alphas_bar_t) + \
            x_t* torch.square(alpha_t)*(1-alphas_bar_t_1)/(1-alphas_bar_t) 
        
        
        # compute predicted 1) variance and 2) mean of p(x_t_1|x_t) by model
        # predicted sigma by model, in equation (15), 
        sigma_model = torch.exp(v*torch.log(betas_t)+ (1-v)*torch.log(sigma_telda))
        sigma_model = sigma_model.clamp(min = 1e-6)     
        sigma_model_log = torch.log(sigma_model)
        # equation (13) from paper, predicted mean by model, denoted as mu_theta
        mean_model = sqrt_recip_alphas_t * (
            x_t - betas_t * noise_pred / sqrt_one_minus_alphas_t
        ) 
        
        # calculate the KL divergence loss between q(x_t_1|x_t, x_0) and p(x_t_1|x_t)
        KL = torch.tensor(KL_0, dtype = torch.float, requires_grad=True).to(device)
        
        for i, t_0 in enumerate(t-1):
            if t_0 != 0:
                KL_i= torch.sum(self.KL_loss(mean_model[i], mean_telda[i], sigma_model_log[i], sigma_telda_log[i]))
                KL_list[i] = KL_i
                KL = KL + KL_i
                
        KL =  KL.sum()/x_0.shape[0] # take average over batch size
        loss = F.l1_loss(noise, noise_pred, reduction = 'sum')/x_0.shape[0]
        # we set weight of KL as 0.1 but different values are possible
        loss = loss + 0.05*KL
        #loss = F.huber_loss(noise, noise_pred, reduction = 'sum')/x_0.shape[0]
        #loss = F.smooth_l1_loss(noise, noise_pred, reduction = 'sum')/x_0.shape[0]
        # we need KL_list to do 'important sampling'
        loss_list = KL_list 
        #torch.norm( noise - noise_pred,p=1, dim=(-2,-1))
        #loss_list = torch.squeeze(loss_list) + KL_list

        return loss, loss_list, KL

    @torch.no_grad()
    def sample_data(self,x, cond, t):
        '''
        sampling data by reverse processs
        get x_t from x_t-1
        Parameters
        ----------
        x : x_t, torch tensor [batch size x channels x height x width]
            image data at time step t
        t : torch tensor array [batch_size]
            time steps where corresponding images are reversed

        Returns
        -------
        TYPE
            x_t: denoised data at time t from t-1

        '''

        device = self.device
        betas_t = self.betas[t]

        # sample sqrt_one_minus_alphas corresponding to time steps
        sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas_bar[t]
        # sample reciprocal alphas corresponding to time steps
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        # sample the sigams corresponding to time steps (posterior variance)
        sigma_t = self.sigmas[t]#[:,None][:,None].to(device)
        # epsilon_theta from model
        noise_pred = self.model(x, cond[:,0], cond[:,1], t)
        # model also outputs variable v to interpolate between beta_t and beta_telda_t
        noise_pred, v = torch.split(noise_pred, 1, dim = 1)
        # sigma value learned during training
        sigma_t_interpolated = torch.exp(v*torch.log(betas_t) + (1-v)*torch.log(sigma_t))
        # mean learned during training
        # reparamaterize the output of model accorting to equation (11)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / sqrt_one_minus_alphas_t
        )
        
        if not torch.is_tensor(t) and t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x).to(device)
            return  model_mean + torch.sqrt(sigma_t_interpolated) * noise

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
        self.model.train()
        
    @torch.no_grad()
    def sample_dataset(self, n_img:int=10, total_timestep:int = 150, cond = None,   img_size:tuple() = None):
        # Sample dataset by n_img from model
        img = torch.randn((n_img, 1, img_size[0], img_size[1]), device=self.device)
        
        for i in range(0,total_timestep)[::-1]:
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            img = self.sample_data(img, cond, t)
            
        # take one sample from repeated values
        
        #path_loss = torch.squeeze(img[:,:,41,:])[:,:25]
        pl = img[:,:,np.arange(8),:].cpu()
        #p_1 = pl[:,:,:, np.arange(50, step =2)]
        p_2 = pl[:,:,:, np.arange(50, step =2)+1]
        #p_3 = torch.concat([p_1,p_2],dim = -2)
        #path_loss = torch.mean(p_2, dim = -2)
        path_loss=  p_2[:,:,0,:]
        return path_loss
    
    @staticmethod
    def KL_loss(mean1, mean2, logvar1, logvar2):
        return 0.5 * (
           -1.0
           + logvar2
           - logvar1
           + torch.exp(logvar1 - logvar2)
           + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
       )
    @staticmethod
    def increase_dim(tensor, n_dim):
        return tensor[(..., ) + (None, ) * n_dim]
 