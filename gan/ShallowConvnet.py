# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:29:05 2022

@author: seongjoon kang
"""
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader

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
        nn.Conv2d(in_channels = input_channels, out_channels = 8, kernel_size=4,stride = 2,  padding=1), #28*25
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(8),
        
        nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=4,stride = 2,  padding=1), #14*12
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
        nn.Linear(100, num_classes)
  
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