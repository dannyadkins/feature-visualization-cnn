import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import dataloader

from cnn import CNN

# small toy model 
hyperparams = {
    'batch_size': 64,
    'test_batch_size': 1000,
    'epochs': 10,
    'lr': 0.01,
    'momentum': 0.5,
    'no_cuda': True,
    'seed': 1,
    'log_interval': 100,
    'save_model': True
}

# load data
train_loader = dataloader.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)) # mean, std
                    ])),
    batch_size=hyperparams['batch_size'], shuffle=True)

test_loader = dataloader.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=hyperparams['test_batch_size'], shuffle=True)