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
from beta_scheduler import  linear_beta_schedule, biquadratic_beta_schedule
from utill import Dataset
from diffusion_model import diffusion_model
def compute_LOS_prob(los_prob, dist_2d):
    
    LOS_count = np.zeros(int(max(dist_2d))+2)
    NLOS_count = np.zeros(int(max(dist_2d))+2)
    dist_2d = np.array(dist_2d/10, dtype = int)
    for j, d_2d_i in enumerate(dist_2d):
        if los_prob[j]>0:
            LOS_count[d_2d_i] +=1
        else:
            NLOS_count[d_2d_i] +=1
    los_prob = LOS_count/(LOS_count + NLOS_count)
    los_prob[0] = 1.0
    return los_prob
            
device = "cuda" if torch.cuda.is_available() else "cpu"
total_timesteps = 150
beta_max = 0.08
betas = linear_beta_schedule(timesteps=total_timesteps, end = beta_max)
#betas = biquadratic_beta_schedule(total_time_steps, end = beta_max)
with open('data_all_1.pickle','rb') as f:
    dataset_1 = pickle.load(f)

with open('data_all_2.pickle','rb') as f:
    dataset_2 = pickle.load(f)

with open('loc_all.pickle','rb') as f:
    loc_all = pickle.load(f)
#lt = ['r-','b:','k-.','g.']
#los_prob_dict=dict()
#for i, h in enumerate([30, 60, 90, 120]):

dataset = np.append(dataset_1, dataset_2, axis =0)


p=0
pl = dataset[:,:,np.arange(8)+8*p,:]
p_1 = pl[:,:,:, np.arange(50, step =2)]
p_2 = pl[:,:,:, np.arange(50, step =2)+1]

pl = (p_2[:,:,p,:]+1)*77
pl = pl.reshape(-1)
model_name = 'diffusion_model_completed.pt'
#model_name = 'diffusion_model_hpc.pt'

model = ResidualAttentionUnet()
model.load_state_dict(torch.load('save_model/%s'%model_name)['model'])
print("Num params: ", sum(p.numel() for p in model.parameters()))
model.to(device)

diffusion_model = diffusion_model (model = model.eval(), total_timesteps=total_timesteps, 
                           beta_max = beta_max, device = device, beta_schedule = linear_beta_schedule)


#dist_2d = np.random.uniform(low = 0, high = 1300, size = 1000)
h = 90
sample_size = 8000
dist_2d_model = np.linspace(0, 1300, sample_size)
h_t = np.random.choice([h], sample_size)
cond_t = torch.tensor(np.column_stack((dist_2d_model, h_t)), dtype = torch.float)
cond_t = cond_t.to(device)

img = diffusion_model.sample_dataset(n_img = sample_size, total_timestep = total_timesteps, 
                                       cond = cond_t,
                                       img_size = dataset.shape[2:])

#path_loss = torch.squeeze(img[:,:,41,:])[:,:25]
pl_model = img[:,:,np.arange(8),:].cpu()
#p_1 = pl[:,:,:, np.arange(50, step =2)]
p_2 = pl_model[:,:,:, np.arange(50, step =2)+1]
#p_3 = torch.concat([p_1,p_2],dim = -2)
#path_loss = torch.mean(p_2, dim = -2)
path_loss=  p_2[:,:,6,:]

#pl_model = pl_model[:,:,:25]*160
pl_model = (path_loss.reshape(-1)+1)*77
pl_model = pl_model.detach().cpu().numpy()


I = np.where(loc_all[:,1]==h)[0]
loc_h = loc_all[I]

dataset_I = dataset[I]
link_state_data = np.squeeze(dataset[I][:,:,55,:][:,:,0])
LOS_prob_data = compute_LOS_prob(link_state_data, loc_h[:,0])


link_state = img[:,:,55,:].cpu() # get LOS state
los_prob_model = torch.squeeze(link_state[:,:,0])
#dist_2d_model = np.array(dist_2d_model/10, dtype = int)
LOS_prob_model = compute_LOS_prob(los_prob_model, dist_2d_model)
plt.scatter(np.arange(len(LOS_prob_model)*10, step = 10), LOS_prob_model, label = 'model')
#plt.figure()
#plt.scatter(dist_2d, los_prob[:,0])
plt.plot(np.arange(len(LOS_prob_data)*10, step = 10), LOS_prob_data,'r',lw =3,  label = 'data')
plt.xlim([-0.1, 1300])
plt.legend()
plt.grid()
plt.xlabel('2D distance (m)')
plt.ylabel ('LOS probability')
plt.title('LOS probability at height = %dm'%h)
plt.savefig('los_prob_%dm'%h)

plt.figure()

plt.plot(np.sort(pl), np.linspace(0,1,len(pl)), label = 'data')
plt.plot(np.sort(pl_model), np.linspace(0,1, len(pl_model)), label = 'model')
plt.grid()
plt.legend()
plt.show()
