# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:56:09 2022

@author: seongjoon kang

"""
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

class ShallowConvnet(nn.Module):
    def __init__(self, input_channels, num_classes):
        """

        Parameters
        ----------
        input_channels : Number of input channels
        num_classes : Number of classes for the final prediction 
        """
        super().__init__()
        self.shallownet = nn.Sequential(
        nn.Conv2d(in_channels = input_channels, out_channels = 8, kernel_size=4,stride = 2,  padding=1), #32*25
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(8),
        
        nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=4,stride = 2,  padding=1), #15*12
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(16),
        
        nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=4,stride = 2,  padding=1), #7*6
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(32),

        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=4,stride = 2,  padding=1), #3*3
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(64),

        #nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=4,stride = 2,  padding=1), #1*1
        #nn.ReLU(),
        #nn.BatchNorm2d(64),

        nn.Flatten(),
        nn.Linear(64*3*3,100),
        nn.LeakyReLU(0.3),
        nn.Linear(100, num_classes),
        #nn.Softmax()
  
        )
        

    def forward(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------
        output : Result after running through the model
        """

        output = self.shallownet(x)
        #print(output.shape)
        #output = output.reshape(-1)
        #output = self.fc(output)
        return output
    
class MyDataSet(Dataset):
  def __init__(self, image, target):
    self.image = image
    self.target = target
  def __len__(self):
    return len(self.target)
  def __getitem__(self, idx):
    return self.image[idx], self.target[idx]

def get_embedding_idx(dist_2D):
    idx_list = np.zeros_like(dist_2D)
    for j, i in enumerate(np.arange(1300, step = 100)):
        I = np.where((dist_2D >=i) & (dist_2D<i+100))[0]
        idx_list[I] = j
    
    I = np.where(dist_2D >1400)[0]
    idx_list[I] = j+1
    
    return torch.LongTensor(idx_list)

if __name__ == "__main__":
    
    with open('data_all_1.pickle','rb') as f:
        dataset_1 = pickle.load(f)
    
    with open('data_all_2.pickle','rb') as f:
        dataset_2 = pickle.load(f)
        
    dataset = np.append(dataset_1, dataset_2, axis =0)
    
    total_data = torch.tensor(dataset, dtype = torch.float32)
    batch_size = 64
    train_size = int(0.8*len(total_data))
    
    with open('loc_all.pickle','rb') as f:
        loc_all = pickle.load(f)
        
    #loc_all = 
    #total_label = torch.tensor(loc_all[:,0], dtype = torch.float32)
    
    total_label = get_embedding_idx(loc_all[:,0])
    
    train_data_set = MyDataSet(image = total_data[:train_size], target= total_label[:train_size])
    test_data_set = MyDataSet(image = total_data[train_size:], target= total_label[train_size:])
    
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True,drop_last=False )
    test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True,drop_last=False )
    
    print(train_data_set.image.shape, train_data_set.target.shape)
    
    model = ShallowConvnet(input_channels=1, num_classes=14)
    model.to (device)
    
    max_patience = 100
    def train_loop(model, criterion, optimizer,  train_loader, val_loader, n_epoch=50):
        """
        Generic training loop
    
        Parameters
        ----------
        model : Object instance of your model class 
        criterion : Loss function 
        optimizer : Instance of optimizer class of your choice 
        train_loader : Training data loader 
        val_loader : Validation data loader
    
        Returns
        -------
        train_losses : List with train loss on dataset per epoch
        train_accuracies : List with train accuracy on dataset per epoch
        val_losses : List with validation loss on dataset per epoch
        val_accuracies : List with validation accuracy on dataset per epoch
    
        """
        best_val = 0.0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
    
        # Training
        for t in tqdm(range(n_epoch)):
          epoch_t_acc = 0.0 
          epoch_t_loss = 0.0
          # Set the model to train mode
          #model.to(device)       
          model.train()
          # Loop over the training set
          for train_data, targets in train_loader:
            # Put the inputs and targets on the write device
            train_data = train_data.to(device)
            targets = targets.to (device)
            # Feed forward to get the logits
            y_pred = model.forward(train_data)
           
            score, predicted = torch.max(y_pred, 1)
            
            # Compute the loss and accuracy
            loss = criterion ( y_pred,targets) 
            # zero the gradients before running
            # the backward pass.
            optimizer.zero_grad()
            # Backward pass to compute the gradient
            # of loss w.r.t our learnable params. 
            loss.backward()
            optimizer.step()
      
            train_accuracy = (targets == predicted).sum().float() / len(targets)
    
            epoch_t_acc += train_accuracy
            epoch_t_loss += loss.item()
    
          train_losses.append(epoch_t_loss/len(train_loader))
          train_accuracies.append(epoch_t_acc/len(train_loader))
    
            # Switch the model to eval modea
          model.eval()
    
          v_acc = 0
          v_loss = 0  
          with torch.no_grad():
              # TLoop over the validation set 
              for val_data, targets in val_loader:
                  # Put the inputs and targets on the write device
                  val_data = val_data.to(device)
                  targets = targets.to(device)
                  # Feed forward to get the logits
                  y_pred_val = model (val_data)
                  score, predicted = torch.max(y_pred_val, 1)
                  # Compute the loss and accuracy
                  loss_val = criterion ( y_pred_val,targets)
                  accuracy_val = (targets == predicted).sum().float() / len(targets)
                  # Keep track of accuracy and loss
                  v_acc += accuracy_val
                  v_loss += loss_val
    
          val_losses.append(v_loss/len(val_loader))
          val_accuracies.append(v_acc/len(val_loader))
    
          if val_accuracies[-1] > best_val:
            best_val = val_accuracies[-1]
            patience_counter = 0
    
            # Save best model, optimizer, epoch_number
            torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': t,
                    }, 'checkpoint.pth')
          else:
            patience_counter += 1    
            if patience_counter > max_patience: 
              break
          print("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACCURACY]: %.3f" % (t, train_losses[-1], train_accuracies[-1]))
          print("[EPOCH]: %i, [VAL LOSS]: %.6f, [VAL ACCURACY]: %.3f \n" % (t, val_losses[-1] ,val_accuracies[-1]))
    
        return train_losses, train_accuracies, val_losses, val_accuracies
    
    
    # Initialize the criterion
    criterion = torch.nn.CrossEntropyLoss()
    # Initialize the SGD optimizer with lr 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Run the training loop using this model
    train_losses, train_accuracies, val_losses, val_accuracies = train_loop(model, criterion, optimizer,  train_data_loader, test_data_loader,100)
    
