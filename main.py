
import torch
import torchvision
import torchvision.transforms as T
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.laplace import Laplace

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pickle as pkl
from pathlib import Path
from functools import partial
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)

import gym
import gym_autotrain

from gym_autotrain.envs.autotrain_env import StateLinkedList

logger = logging.getLogger(__file__)

# data

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def splitds(train, test, no_signal=False):
    X = np.concatenate((train.data,test.data), axis=0)
    Y = train.targets + test.targets
    
    if no_signal:
        print('suffling labels')
        np.random.shuffle(Y)
    
    split_id = int(len(X) * DATA_SPLIT)
    train.data, train.targets = X[:split_id], Y[:split_id]
    test.data, test.targets = X[split_id:], Y[split_id:]

def get_dataset(tfms, no_signal=False):
    trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT / 'cifar-10-data', train=True,
                                        download=True, transform=tfms)

    testset = torchvision.datasets.CIFAR10(root=DATA_ROOT / 'cifar-10-data', train=False,
                                           download=True, transform=tfms)
    
    splitds(trainset, testset, no_signal)
    
    return trainloader, holdoutloader


# testing with random agent

agent = lambda dim: np.random.rand(dim)

def accuracy(model, data): # phi
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in tqdm(data,total=len(data)):
            images, labels = data[0].to(DEVICE), data[1]
            outputs = model(images).cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total

if __name__ == '__main__':


    # data init
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    
    TFMS = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

    train, holdout = get_dataset(TFMS)
    
    # backboone model definition
    backbone = models.resnet18(pretrained=False)
    
    env = gym.make('AutoTrain-v0')
    env.init(backbone=backbone,  phi: callable, savedir:Path,
             trnds:torchdata.Dataset, valds:torchdata.Dataset, 
             T=3, H=5, K=256, lr_init=3e-4, inter_reward=0.05,
             num_workers=4, bs=16, log_file=None)
    
    logger.info('environment initialised')
    
    # environment & replay interactions
