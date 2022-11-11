# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:56:59 2022

@author: seongjoon kang
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dc_gan import Discriminator, Generator, initialize_weight


device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
LERNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = [64,64]
CHANNELS_IMG = 3

Z_DIM = 100
NUM_EPOCHS =5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
         transforms.Resize(IMAGE_SIZE),
         transforms.ToTensor(),
         transforms.Normalize(
             [0.5 for _ in range(CHANNELS_IMG)], 
             [0.5 for _ in range(CHANNELS_IMG)]),
     ]
    )
#dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
dataset = datasets.ImageFolder(root = 'celeb_dataset', transform = transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weight(gen)
initialize_weight(disc)

opt_gen = optim.Adam(gen.parameters(), lr = LERNING_RATE, betas = (0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr = LERNING_RATE, betas = (0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1,1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, Z_DIM,1,1).to(device)
        fake = gen(noise)
        ##### Train Discrminator max log(D(x)) + log ( 1- D(G(z)))
        
        disc_real = disc(real).reshape(-1) 
        loss_disc_real = criterion (disc_real, torch.ones_like(disc_real))
        
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()
        
        ### Train Generator min log ( 1- D(G(z))) <-> max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        
        
        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            disc.eval()
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise)
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
            disc.train()

        


