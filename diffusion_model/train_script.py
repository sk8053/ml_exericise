# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:57:20 2022

@author: seongjoon kang
"""

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import Adam, lr_scheduler
import pickle
from diffusion_model import diffusion_model
from ResidualAttentionUnet import ResidualAttentionUnet
#from beta_scheduler import linear_beta_schedule
from utill import Dataset, show_images, show_tensor_image
#from beta_scheduler import sigmoid_beta_schedule, cosine_beta_schedule, biquadratic_beta_schedule
from beta_scheduler import linear_beta_schedule, multi_linear_schedule

total_timesteps =150
beta_max = 0.08
lr = 1e-3
gamma = 0.9
BATCH_SIZE = 64
epochs = 1000 # Try more!
beta_schedule = linear_beta_schedule
#betas = linear_beta_schedule(timesteps=total_timesteps, end = beta_max)
#betas = semi_linear_scheduler(timesteps=total_timesteps, end = beta_max)
load_model = False
device = "cuda" if torch.cuda.is_available() else "cpu"

with open('data_all_1.pickle','rb') as f:
    dataset_1 = pickle.load(f)

with open('data_all_2.pickle','rb') as f:
    dataset_2 = pickle.load(f)

with open('loc_all.pickle','rb') as f:
    loc_all = pickle.load(f)
print('cond variable shape,', loc_all.shape)    
dataset = np.append(dataset_1, dataset_2, axis =0)
print('data loaded ', dataset.shape)
dataset = np.append(dataset_1, dataset_2, axis =0)
pl = dataset[:,:,np.arange(8),:]
#p_1 = pl[:,:,:, np.arange(50, step =2)]
p_2 = pl[:,:,:, np.arange(50, step =2)+1]
pl = (p_2[:,:,0,:]+1)*77
pl = pl.reshape(-1) # pathloss from data

data= torch.tensor(dataset,dtype= torch.float) # make the range of data as [-1, 2]

#A = 10*np.array([[4,3,2,1,-1,-2,-3,-4],[5,4,3,2,-2,-3,-4,-5]])/160
#B = np.repeat(A[None], 350, axis =0 )
#B = B.reshape(-1, 56, 50)[None]
# diversify the repeated data to get diversity gain after sampling
#data = data + torch.tensor(B, dtype = torch.float32)
dataset_with_cond = Dataset(dataset, loc_all)  
dataloader = DataLoader(dataset_with_cond, batch_size=BATCH_SIZE, shuffle=True)


model = ResidualAttentionUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

if load_model == True:
    model.load_state_dict(torch.load('save_model/diffusion_model.pt')['model'])
    optimizer.load_state_dict(torch.load('save_model/diffusion_model.pt')['optz_state'])
    min_loss = torch.load('save_model/diffusion_model.pt')['loss']
    print(f'minimum loss is {min_loss}')
else: 
    min_loss = 2e5
diffusion_model = diffusion_model (model = model.eval(), total_timesteps=total_timesteps, 
                           beta_max = beta_max, device = device, beta_schedule = beta_schedule)

# remove the below part in hpc
show_images(data)
# Simulate forward diffusion
# show images after several diffusions
image = next(iter(dataloader))[0]
plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(total_timesteps/num_images)
for idx in range(0, total_timesteps, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    image, noise = diffusion_model.forward_data(image, t)
    show_tensor_image(image.detach().cpu().numpy(), idx)
    
plt.colorbar()
plt.show()

loss_avg = 0
loss_table = np.ones((total_timesteps,10))*10

for epoch in range(epochs):
    for step, (batch, cond) in enumerate(dataloader):
      model.train()
      curr_batch_size = batch.shape[0]
      batch = batch.to(device, dtype = torch.float)
      cond = cond.to (device, dtype = torch.float)
      
      optimizer.zero_grad()
      
      t = torch.randint(0, total_timesteps, (curr_batch_size,), device=device).long()
      if step % 3 == 0:
          #uniformly sample time steps 
          t = torch.randint(0, total_timesteps, (curr_batch_size,), device=device).long()
      else:
          loss_sum = loss_table.sum(1)
          loss_prob = loss_sum / np.sum(loss_sum)
          loss_prob = torch.Tensor(loss_prob).to(device)
          #sample time steps based on the loss values 
          t = torch.multinomial(loss_prob, curr_batch_size, replacement =True).to(device)
          
      loss, loss_weights, KL = diffusion_model.get_loss(model, batch, cond, t)
      
      save_model_name = 'diffusion_model_loss_%d_KL_%d'%(int(loss), int(KL))
      loss_weights = loss_weights.detach().cpu().numpy()
      t_ = t.detach().cpu().numpy()
      loss_table[t_, step%10] = loss_weights
      
      loss.backward()
      optimizer.step()
      loss_avg += loss.item()
      
    loss_avg = loss_avg/len(dataloader)  
    scheduler.step()
    if epoch % 2 == 0:
      print(f"Epoch {epoch} | step {step:03d} Loss: {loss_avg} KL Loss: {KL}")
      # remove the below line in hpc
      diffusion_model.model.eval()
      diffusion_model.sample_plot_image(dataset.shape[2:], cond[0][None])
      diffusion_model.model.train()
      
    if loss_avg < min_loss:
        torch.save({'model':model.state_dict(),
                    'loss':loss_avg,
                    'epoch': epoch,
                    'optz_state':optimizer.state_dict(),
                    },
                   'save_model/%s'%save_model_name)
    
        print('epoch,', epoch, 
                'model is saved and loss is ', loss_avg)
        min_loss = loss_avg
    # remove the below parts in hpc    
    if epoch % 5 ==0: # compare distribution of pathloss per every 5 epoch
        dist_2d = np.random.uniform(low = 0, high = 1300, size = 500)
        h = np.random.choice([30,60,90,120], 500)
        cond_t = torch.tensor(np.column_stack((dist_2d, h)), dtype = torch.float)
        cond_t = cond_t.to(device)
        diffusion_model.model.eval()
        img = diffusion_model.sample_dataset(n_img = 500, total_timestep = total_timesteps, 
                                               cond = cond_t,
                                               img_size = dataset.shape[2:])
        diffusion_model.model.train()
        #path_loss = torch.squeeze(img[:,:,41,:])[:,:25]
        pl = img[:,:,np.arange(8),:].cpu()
        #p_1 = pl[:,:,:, np.arange(50, step =2)]
        p_2 = pl[:,:,:, np.arange(50, step =2)+1]
        #p_3 = torch.concat([p_1,p_2],dim = -2)
        #path_loss = torch.mean(p_2, dim = -2)
        pl_model_ =  p_2[:,:,0,:]
        
        pl_model = (pl_model_.reshape(-1)+1)*77
        pl_model = pl_model.detach().cpu().numpy()

        plt.figure()
        plt.plot(np.sort(pl), np.linspace(0,1,len(pl)),'k', label = 'data')
        plt.plot(np.sort(pl_model), np.linspace(0,1, len(pl_model)),'r', label = 'model')
        plt.grid()
        plt.legend()
        plt.show()
        
    loss_avg = 0