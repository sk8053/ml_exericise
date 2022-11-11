# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:45:53 2022

@author: gangs
"""
import torch
import matplotlib.pyplot as plt

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

# some useful function to show images
def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])
    plt.colorbar()
def show_tensor_image(image,i):   
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(image[0])
    plt.title('t = %d'%i)
