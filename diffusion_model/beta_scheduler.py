# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 08:21:33 2022

@author: seongjoon kang
"""
import torch


def cosine_beta_schedule(timesteps, s=0.008, end = 0.02):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas*end
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, end = 0.02):
    beta_start = 0.0001
    beta_end = end
    return torch.linspace(beta_start, beta_end, timesteps)

def biquadratic_beta_schedule(timesteps, end = 0.02):
    beta_start = 0.0001
    beta_end = end
    return torch.linspace(beta_start**2, beta_end**2, timesteps) ** 0.5

def sigmoid_beta_schedule(timesteps, end = 0.02):
    beta_start = 0.0001
    beta_end = end
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def multi_linear_schedule (timesteps=150, end=0.02):
  half1 = int(3*timesteps/4)
  half2 = timesteps - half1
  beta1 = torch.linspace(0.00001, end/10, half1)
  beta2 = torch.linspace(end/10, end, half2)
  betas = torch.concat([beta1, beta2])

  return betas