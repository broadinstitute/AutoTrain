import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from pathlib import Path

import autotrain.agent.A2C.utils as utils
import autotrain.agent.A2C.model as model
from autotrain.envs.autotrain_env import AutoTrainEnvironment

from itertools import count

import time, gc

GAMMA = 0.99
TAU = 0.001


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Trainer:

    def __init__(self, env: AutoTrainEnvironment, state_shape: np.array, action_dim: int,
                 action_lim: np.array, ram, savedir: Path, bs=32, lr=3e-4, dev=None):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :return:
        """
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.action_lim = action_lim  #  numpy; action limiting is to be done by the agent

        self.env = env
        self.ram = ram
        self.iter = 0
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.savedir = savedir

        self.bs = bs
        self.lr = lr
        self.dev = dev

        self.actor = model.Actor(self.state_shape, self.action_dim)
        self.target_actor = model.Actor(self.state_shape, self.action_dim)

        self.critic = model.Critic(self.state_shape, self.action_dim)
        self.target_critic = model.Critic(self.state_shape, self.action_dim)

        if self.dev:
            self.actor.to(self.dev)
            self.target_actor.to(self.dev)

            self.critic.to(self.dev)
            self.target_critic.to(self.dev)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.lr)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        shape = state.shape
        state = torch.FloatTensor(state).view(1, *shape)
        state = Variable(state).to(self.dev) if self.dev else Variable(state)

        action = self.target_actor.forward(state).detach().cpu()
        action = sigmoid(action.numpy()) * self.action_lim
        return action

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        shape = state.shape
        state = torch.FloatTensor(state).view(1, *shape)
        state = Variable(state).to(self.dev) if self.dev else Variable(state)

        action = self.actor.forward(state).detach().cpu()
        new_action = action.data.numpy() + self.noise.sample()  # oops
        new_action = sigmoid(new_action) * self.action_lim
        return new_action[0]

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        if self.ram.len < self.bs:
            return 
        
        s1, a1, r1, s2 = self.ram.sample(self.bs)

        s1 = Variable(torch.from_numpy(s1)).to(self.dev) if self.dev else Variable(torch.from_numpy(s1))
        a1 = Variable(torch.from_numpy(a1)).to(self.dev) if self.dev else Variable(torch.from_numpy(s1))
        r1 = Variable(torch.from_numpy(r1)).to(self.dev) if self.dev else Variable(torch.from_numpy(s1))
        s2 = Variable(torch.from_numpy(s2)).to(self.dev) if self.dev else Variable(torch.from_numpy(s1))

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r1 + GAMMA * next_val
        # y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        loss_actor = -1 * torch.sum(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        utils.soft_update(self.target_actor, self.actor, TAU)
        utils.soft_update(self.target_critic, self.critic, TAU)
        print(f'[ATA] optimization step performed')

    def episode(self, i):
        observation, _ = self.env.reset()

        for t in count():
            start_time = time.time()

            action = self.get_exploration_action(observation)

            new_observation, reward, done, info = self.env.step(action)
            
            
            if done:
                
                if info.result[-1] > self.env._baseline.result[-1]:
                    self.env.set_baseline(info)
                    
                self.ram.add(observation, action, reward, np.zeros(self.env.observation_space.shape))
                break
                
            self.ram.add(observation, action, reward, new_observation)

            new_observation[new_observation == 255] = 0

            observation = new_observation

            # perform optimization
            self.optimize()

            print(f'[ATA episode {i}]: took [{time.time() - start_time:.1f}] seconds for one full step')


        gc.collect()

        # save env, buffer, agent
        episode_dir = self.savedir / f'{i}_episode'
        episode_dir.mkdir(exist_ok=True)

        self.env.save(episode_dir)
        self.ram.save(self.savedir / 'mem.pkl')
        self.save(episode_dir)

        return t

    def save(self, savedir):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), savedir / f"actor.pt")
        torch.save(self.target_critic.state_dict(), savedir / f"critic.pt")

    def load(self, savedir):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load(savedir / f"actor.pt"))
        self.critic.load_state_dict(torch.load(savedir / f"critic.pt"))

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
