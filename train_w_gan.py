"""
Training of DCGAN network with WGAN loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from w_gan import Discriminator, Generator, initialize_weight
from utill import gradient_penalty
import numpy as np
import pickle
# Hyperparameters etc
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = [64, 64]
CHANNELS_IMG = 1
Z_DIM = 10
NUM_EPOCHS = 5
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_gp= 10
#WEIGHT_CLIP = 0.01


with open('data_all.pickle','rb') as f:
    dataset = pickle.load(f)
print('data is loaded')
with open('loc_all.pickle','rb') as f:
    loc_all = pickle.load(f)
    
dataset= torch.tensor(dataset)
loc_all = torch.tensor(loc_all)


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        # Load data and get label    
        y = self.labels[index]

        return X, y
    
data_set_ob = Dataset(dataset, loc_all)
loader = DataLoader(data_set_ob, batch_size=BATCH_SIZE, shuffle=True)

# initialize gen and disc/critic
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weight(gen)
initialize_weight(critic)

# initializate optimizer
#opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
#opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE,
                     betas = (0.0,0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE,
                        betas = (0.0,0.9))

# for tensorboard plotting
#fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, cond) in enumerate(loader):
     
        cond[:,1] = torch.Tensor(np.random.uniform(low = cond[:,0], high=cond[:,1]))
        
        new_cond = cond[:,[1,2]]     
        old_cond = cond[:,[0,2]]
        
        cond = old_cond.to(device, dtype = torch.float)
        new_cond = new_cond.to(device, dtype = torch.float)
        real = real.to(device, dtype = torch.float)
        
        cur_batch_size = real.shape[0]
       
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, cond)
      
            critic_real = critic(real, cond).reshape(-1)
            critic_fake = critic(fake, cond).reshape(-1)
            gp = gradient_penalty(critic, real, fake,cond, device = device)
            loss_critic =( -(torch.mean(critic_real) - torch.mean(critic_fake))
            + LAMBDA_gp*gp)
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()


        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, cond).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, new_cond)
      
            critic_real = critic(real, new_cond).reshape(-1)
            critic_fake = critic(fake, new_cond).reshape(-1)
            gp = gradient_penalty(critic, real, fake,new_cond, device = device)
            loss_critic =( -(torch.mean(critic_real) - torch.mean(critic_fake))
            + LAMBDA_gp*gp)
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # clip critic weights between -0.01, 0.01
            #for p in critic.parameters():
            #    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, new_cond).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
       
        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            critic.eval()
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise, cond)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            gen.train()
            critic.train()

torch.save(gen.state_dict(), 'save_model/gen.pt')
torch.save(critic.state_dict(), 'save_model/disc.pt')
