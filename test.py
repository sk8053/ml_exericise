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
Z_DIM = 10
#NUM_EPOCHS = 5
FEATURES_CRITIC = 64
FEATURES_GEN = 64
SAMPLE_SIZE = 2000

#checkpoint = torch.load('save_model/gen_first_trial.pt')
checkpoint = torch.load('save_model/cond_gen_second.pt')

model = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)
model = model.to(device)
model.load_state_dict(checkpoint)

with open('data_all.pickle','rb') as f:
    dataset = pickle.load(f)
print(dataset.shape)

noise = torch.randn(SAMPLE_SIZE, Z_DIM, 1, 1).to(device)
dist_2d = np.random.uniform(low = 600, high = 1200, size = SAMPLE_SIZE)
h = np.repeat([60],SAMPLE_SIZE)
cond = np.column_stack((dist_2d, h))
cond = torch.Tensor(cond)
predicted= model(noise,cond)

path_loss = predicted[:,:,1,:]*16 +87
pl_min = torch.min(path_loss, dim = -1)[0]
plt.scatter(cond[:,0].cpu().detach().numpy(), pl_min.cpu().detach().numpy())



