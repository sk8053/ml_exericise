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
import matplotlib.pyplot as plt
from train_estimator import ShallowConvnet
from train_estimator_for_height import ResidualConvnet

# Hyperparameters etc
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
IMAGE_SIZE = [64, 64]
CHANNELS_IMG = 1
Z_DIM = 5
NUM_EPOCHS = 150
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_gp= 10
#WEIGHT_CLIP = 0.01


with open('data_all_1.pickle','rb') as f:
    dataset_1 = pickle.load(f)

with open('data_all_2.pickle','rb') as f:
    dataset_2 = pickle.load(f)
    
dataset = np.append(dataset_1, dataset_2, axis =0)
print ('data is loaded', dataset.shape)
with open('loc_all.pickle','rb') as f:
    loc_all = pickle.load(f)
    
dataset= torch.tensor(dataset)
loc_all = torch.tensor(loc_all)
print(loc_all.shape)

model = ShallowConvnet(1, num_classes=14)
model.load_state_dict(torch.load('checkpoint.pth')['model'])
model.to(device)
model.eval()

model_h = ResidualConvnet(1, num_classes=4)
model_h.load_state_dict(torch.load('checkpoint_height.pth')['model'])
model_h.to(device)
model_h.eval()

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
    
def get_embedding_idx(dist_2D):
    idx_list = np.zeros_like(dist_2D)
    for j, i in enumerate(np.arange(1300, step = 100)):
        I = np.where((dist_2D >=i) & (dist_2D<i+100))[0]
        idx_list[I] = j
    
    I = np.where(dist_2D >1400)[0]
    idx_list[I] = j+1
    
    return torch.LongTensor(idx_list)
    
    
Embbed_book = torch.nn.Embedding(15,6)
Embbed_book2 = torch.nn.Embedding(4,3)
n_cond = 9
torch.save(Embbed_book, 'embedding_book.pt')
torch.save(Embbed_book2, 'embedding_book2.pt')

discrete_dist2d = get_embedding_idx(loc_all[:,0].detach().numpy()) 

embbeded_dis2d = Embbed_book(discrete_dist2d)

discrete_height = torch.LongTensor(loc_all[:,1].detach().numpy()/30-1)
embbeded_height = Embbed_book2(discrete_height)

loc_all = torch.concat([embbeded_height, embbeded_dis2d, discrete_dist2d[:,None], discrete_height[:,None]],dim=1)
loc_all = torch.Tensor(loc_all.detach().numpy())

data_set_ob = Dataset(dataset, loc_all)
loader = DataLoader(data_set_ob, batch_size=BATCH_SIZE, shuffle=True)

# initialize gen and disc/critic
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)

initialize_weight(gen)
initialize_weight(critic)

#gen.load_state_dict(torch.load('save_model/gen.pt'))
#critic.load_state_dict(torch.load('save_model/disc.pt'))
# initializate optimizer

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE,
                     betas = (0.0,0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE,
                        betas = (0.0,0.9))

# for tensorboard plotting
#fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
#writer_cond_ch = SummaryWriter(f"logs/cond_check")
step = 0

gen.train()
critic.train()
criterion = torch.nn.CrossEntropyLoss()
best_loss_critic = -1e5
best_loss_gen = 1e5

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, cond) in enumerate(loader):
        
        cond0 = torch.LongTensor(cond[:,n_cond].detach().cpu().numpy())
        cond_h = torch.LongTensor(cond[:,n_cond+1].detach().cpu().numpy())
        
        cond = cond[:,:n_cond]
        cond = cond.to(device, dtype = torch.float)
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
   
        loss_cond_2d, loss_cond_h = 0, 0
        train_accuracy_2d, train_accuracy_h = 0, 0
        
        if epoch ==0: 
            n_iter = 1
        else:
            n_iter=10
            
        for _ in range(n_iter):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, cond)
            
            y = model(fake)
            _, y_pred = torch.max(y, 1)
            cond0 = cond0.to(device)
            loss_cond_2d += criterion ( y ,cond0)
            train_accuracy_2d += (cond0 == y_pred).sum().float() / len(cond0)
            
            y_h = model_h(fake)
            _, y_pred_h = torch.max(y_h, 1)
            cond_h = cond_h.to(device)
            loss_cond_h += criterion ( y_h ,cond_h)
            train_accuracy_h += (cond_h == y_pred_h).sum().float() / len(cond_h)
        
        loss_cond_2d = loss_cond_2d/n_iter
        loss_cond_h = loss_cond_h/n_iter
        train_accuracy_2d = train_accuracy_2d/n_iter
        train_accuracy_h = train_accuracy_h/n_iter

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, cond).reshape(-1)
        loss_gen = -torch.mean(gen_fake) + loss_cond_2d + loss_cond_h
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        if loss_critic > best_loss_critic and loss_gen < best_loss_gen:
            torch.save(gen.state_dict(), 'save_model/gen.pt')
            torch.save(critic.state_dict(), 'save_model/disc.pt')
            best_loss_critic = loss_critic
            best_loss_gen = loss_gen
            print(f"model is saved, Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")
        # Print losses occasionally and print to tensorboard
        if (batch_idx % 100 == 0 and batch_idx > 0) or batch_idx ==0 :
            gen.eval()
            critic.eval()
            print(f" Epoch [{epoch}/{NUM_EPOCHS}]================= Batch [{batch_idx}/{len(loader)}]===================")
            print(f"Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}, loss cond 2d: {loss_cond_2d:.4f}, loss cond h: {loss_cond_h:.4f}")
            print(f'accuracy 2d: {train_accuracy_2d:.4f}, accuracy h: {train_accuracy_h:.4f}')
            
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
                
                ############### check conditionality ############################
                if batch_idx % 300 == 0:
                    noise_new = torch.randn(500, Z_DIM, 1, 1).to(device)
                    dist_2d = np.random.uniform(low = 300, high = 1300, size = 500)
                    h = np.repeat([60],500)
                    
                    
                    discrete_dist2d = get_embedding_idx(dist_2d)
                    embbeded_dis2d = Embbed_book(discrete_dist2d)
    
                    discrete_height = torch.LongTensor(h/30-1)
                    embbeded_height = Embbed_book2(discrete_height)
    
                    cond_new = torch.concat([embbeded_height, embbeded_dis2d],dim=1)
                    cond_new = cond_new.to(device)
                    #cond_new = torch.Tensor(embbeded_height).to(device)
                    
                    predicted= gen(noise_new,cond_new)
    
                    p1 = predicted[:,:,np.arange(8),:]*160 +87
                    p1 = p1[:,:,:, np.arange(50, step =2)]
                    
                    p2 = predicted[:,:,np.arange(8),:]*160 +87
                    p2 = p2[:,:,:,np.arange(50, step =2)+1]
                    
                    p3 = torch.concat([p1,p2],dim = -2)
                    path_loss = torch.mean(p3, dim = -2)
                    
                    pl_min = torch.min(path_loss, dim = -1)[0]
                    
                    dist_3d = np.sqrt(dist_2d**2 + h**2)
                    fspl = 20*np.log10(np.sort(dist_3d)) + 20*np.log10(28e9) -147.55
                    
                    plt.figure()
                    plt.scatter(dist_3d, pl_min.cpu().detach().numpy())
                    plt.plot(np.sort(dist_3d), fspl, 'r')
                    plt.show()
                    
                    #writer_cond_ch.add_image("cond_check", plt.gcf(), global_step=step)
                    

            step += 1
            gen.train()
            critic.train()


