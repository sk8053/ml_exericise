# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:29:48 2022

@author: gangs
"""
import torch
import numpy as np
import pickle
from w_gan import Discriminator, Generator
import matplotlib.pyplot as plt
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
#LEARNING_RATE = 1e-4
#BATCH_SIZE = 64
#IMAGE_SIZE = [64, 64]
CHANNELS_IMG = 1
Z_DIM = 5
#NUM_EPOCHS = 5
FEATURES_CRITIC = 64
FEATURES_GEN = 64
SAMPLE_SIZE = 2000
   
def get_embedding_idx(dist_2D):
    idx_list = np.zeros_like(dist_2D)
    for j, i in enumerate(np.arange(1300, step = 100)):
        I = np.where((dist_2D >=i) & (dist_2D<i+100))[0]
        idx_list[I] = j
    
    I = np.where(dist_2D >1400)[0]
    idx_list[I] = j+1
    
    return torch.LongTensor(idx_list)
    
#checkpoint = torch.load('save_model/gen_first_trial.pt')
checkpoint = torch.load('save_model/gen.pt')

model = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)
model = model.to(device)
model.load_state_dict(checkpoint)

#with open('data_all.pickle','rb') as f:
#    dataset = pickle.load(f)
#print(dataset.shape)

noise = torch.randn(SAMPLE_SIZE, Z_DIM, 1, 1).to(device)
dist_2d = np.random.uniform(low = 300, high = 1300, size = SAMPLE_SIZE)
h = np.repeat([30],SAMPLE_SIZE)

Embbed_book = torch.load('embedding_book.pt')
Embbed_book2 = torch.load('embedding_book2.pt')

discrete_dist2d = get_embedding_idx(dist_2d)
embbeded_dis2d = Embbed_book(discrete_dist2d)

discrete_height = torch.LongTensor(h/30-1)
embbeded_height = Embbed_book2(discrete_height)


cond = torch.concat([embbeded_height, embbeded_dis2d],dim=1)
cond = torch.Tensor(cond)

predicted= model(noise,cond)
p1 = predicted[:,:,np.arange(8),:]*160 +87
p1 = p1[:,:,:, np.arange(50, step =2)]

p2 = predicted[:,:,np.arange(8),:]*160 +87
p2 = p2[:,:,:,np.arange(50, step =2)+1]

p3 = torch.concat([p1,p2],dim = -2)
path_loss,_ = torch.max(p3, dim = -2)

pl_min = torch.min(path_loss, dim = -1)[0]

plt.figure()
dist_3d = np.sqrt(dist_2d**2 + h**2)

fspl = 20*np.log10(np.sort(dist_3d)) + 20*np.log10(28e9) -147.55
plt.scatter(dist_3d, pl_min.cpu().detach().numpy())
plt.plot(np.sort(dist_3d), fspl, 'r')


