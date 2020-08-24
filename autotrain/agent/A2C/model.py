import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import ceil

EPS = 0.003


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def compute_conv_output(w, h, nl, S=1, F=5, P=0):
    for _ in range(nl):
        w = ceil((w - F + 2 * P) / (S + 1))
        h = ceil((h - F + 2 * P) / (S + 1))
    return w, h


class ConvNet(nn.Module):  # needs to be
    def __init__(self, input_shape):
        # 7, 256, 256
        super(ConvNet, self).__init__()
        nc, width, height = input_shape
        nw, nh = compute_conv_output(width, height, 3)

        self.fc_dim = 8 * nw * nh

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(nc, 32, 5)  #  S=1;P=0;F=5; -- OUTPUT: 32,(256-5)/2,(256-5)/2
        self.conv2 = nn.Conv2d(32, 16, 5)  #  INPUT: 32, 125, 125; -- OUTPUT: 16, (125-5)/2, (125-5)/2
        self.conv3 = nn.Conv2d(16, 8, 5)  #  INPUT: 16, 60, 60; -- OUTPUT: 8,55/2,55/2

        self.fc1 = nn.Linear(self.fc_dim, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, self.fc_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Critic(nn.Module):

    def __init__(self, state_shape, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_shape = state_shape
        self.action_dim = action_dim

        self.state_net = ConvNet(state_shape)

        self.action_net = nn.Sequential(
            nn.Linear(action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.value_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.value_net[-1].weight.data.uniform_(-EPS, EPS)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """

        state_rep = self.state_net(state)
        action_rep = self.action_net(action)

        x = torch.cat((state_rep, action_rep), dim=1)

        x = self.value_net(x)
        return x


class Actor(nn.Module):

    def __init__(self, state_shape, action_dim):
        """
        :param state_shape: Dimension of input state np.array
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim, action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_shape = state_shape
        self.action_dim = action_dim

        self.state_net = ConvNet(self.state_shape)

        self.fc = nn.Linear(128, action_dim)
        self.fc.weight.data.uniform_(-EPS, EPS)

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        x = self.state_net(state)
        action = F.tanh(self.fc(x))

        return action
