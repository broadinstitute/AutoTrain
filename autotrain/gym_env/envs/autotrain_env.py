import gym
from gym import error, spaces
from gym.utils import seeding

import autotrain.gym_env.envs.utils as utils

from autotrain.gym_env.envs.thresholdout import Thresholdout

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as  torchdata

import pandas as pd
import numpy as np

from pathlib import Path
from functools import partial
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt


def make_o(loss_vec: np.array, lr: float, phi_val: float):
    return np.concatenate((loss_vec, [lr, phi_val]), axis=0)


class AutoTrainEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def __repr__(self):
        return f"""AutoTrainEnvironment with the following parameters:
                        lr_init={self.lr_init}, inter_reward={self._inter_reward}, 
                        H={self.H}, K={self.K}, T={self.T}"""

    def log(self, s):
        if self.v: print(f'[time_step:{self.time_step}] ', s)

    def get_env_params(self) -> dict:
        return {
            'K': self.K,
            'T': self.T,
            'H': self.H,
            'sampling_interval': self.sampling_interval,
            'save_dir': str(self.savedir),
            'inter_reward': self._inter_reward,
        }

    def init(self, backbone: nn.Module, phi: callable, savedir: Path,
             trnds: torchdata.Dataset, valds: torchdata.Dataset,
             T=3, H=5, S=2, lr_init=3e-4, inter_reward=0.05, final_reward_scale=10,
             horizon=50, criterion=nn.CrossEntropyLoss(),
             num_workers=4, bs=16, v=False, device=None):
        """
        params:
              - backbone: nn.Module, neural network architecture for training
              - trainds, valds: Dataset, train and validation datasets
              - phi: callable, function to be optimised

              - T: num batch updates that constitutes one time step for the environment
              - H: length of rewind vector
              - S: sampling interval, determines K which is the loss vector
        """
        # experiment paramter setup

        self.T, self.H, self.sampling_interval = T, H, S
        self.K = self.T // self.sampling_interval
        self.horizon = horizon
        self._inter_reward = inter_reward
        self._fr_scale = final_reward_scale
        self.v = v  # is verbose
        self.device = device
        self.bs = bs
        self.num_workers = num_workers
        self.savedir = savedir

        #
        self.action_space_dim = 7
        # loss_vec of size K + lr + phi_val
        self.observation_space_dim = (self.K + 2) * H

        self.time_step = 0

        # model
        self.ll = StateLinkedList(savedir=savedir, dim=self.K + 2)
        self.backbone = backbone

        self.criterion = criterion
        self.lr_init = self._curr_lr = lr_init

        self._init_backbone()

        # package data
        self.trnds, self.valds = trnds, valds

        self.trndl, self.fixdl, self.valdl = utils.create_dls(self.trnds, self.valds, bs=bs, num_workers=num_workers)

        # Thresholdout & statistic of interest
        self.thresholdout = Thresholdout(self.trndl, self.valdl)
        self.phi = phi
        self._phi_func = partial(self.phi, model=self.backbone)  #   does partial work on this

        self._init_phi()

        #  calculate the sampling interval

        self.logmdp = pd.DataFrame(columns=['t', 'phi', 'reward', 'action', 'clf. optim. steps', 'lr'])
        self.logloss = dict()

        self._add_observation(np.zeros(self.K), self._get_phi_val())
        self._append_log(np.zeros(self.K), self._get_phi_val(), "ENV INIT", 0)

        self.log(f'environment initialised : {self.__repr__()}')

        return self.ll.get_observations(self.H)

    def _init_phi(self):
        self.log('initialised phi value: started ...')
        self._prev_phi_val = 0
        self._cur_phi_val = self.thresholdout.verify(self._phi_func)
        self._last_phi_update = self.time_step
        self.log('initialised phi value: done')

    def _get_phi_val(self) -> float:

        if self._last_phi_update != self.time_step:
            self._prev_phi_val = self._cur_phi_val
            self._cur_phi_val = self.thresholdout.verify(self._phi_func)
            self._last_phi_update = self.time_step

        return self._cur_phi_val

    def _init_backbone(self):

        if self.device:
            self.backbone.to(self.device)  #  check

        utils.init_params(self.backbone)

        self.opt = optim.SGD(self.backbone.parameters(), lr=self._curr_lr)
        self.log(f'initialised backbone parameters & optimizer')

    def _get_verb_action(self, action_id: int) -> str:

        if action_id == self.action_space_dim:
            return "STOP signal"
        elif action_id == self.action_space_dim - 1:
            return "RE-INIT signal & keep lr constant"
        elif action_id == 0:
            return 'keep lr constant'
        elif action_id == 1:
            return 'decrease lr by 20%'
        elif action_id == 2:
            return 'decrease lr by 80%'
        elif action_id == 3:
            return 'increase lr by 20%'
        elif action_id == 4:
            return 'increase lr by 80%'

    def _append_log(self, loss_vec: np.array, phival: float, action_vrb: str, reward: float):

        # vrb actions are good, easier to inspect

        self.logmdp.loc[len(self.logmdp)] = [self.time_step, phival, reward, action_vrb, self.ll.len, self._curr_lr]

        self.logloss[self.time_step] = loss_vec

    def save_env(self, savedir: Path):

        assert savedir.is_dir() and savedir.exists()

        self.logmdp.to_csv(savedir / 'env_mdp_log.csv')

        with (savedir / 'env_loss.pkl').open('wb') as fp:
            pkl.dump(self.logloss, fp)

        with (savedir / 'env_params.pkl').open('wb') as fp:
            pkl.dump(self.get_env_params(), fp)

    def _add_observation(self, loss_vec: np.array, phi_val: float):  #  maybe use record log here
        o_state = ObservationAndState(
            param_dict=self.backbone.state_dict(),
            o=make_o(loss_vec, self._curr_lr, phi_val)
        )
        self.ll.append(o_state)
        self.log(f'added observation')

    def visualise_data(self):
        """
        two plots:
              - class distribution of training data
              - class distribution of validation data
        """
        pass

    def step(self, action: int):
        """
        step(self, action: int):
            @action: index of the max value of the action probability vector
        action vector:
            0 - keep constant
            1 - decrease 20%
            2 - decrease 80%
            3 - increase 20%
            4 - increase 80%
            5 - re-init & keep constant
            6 - stop
        """

        self.log(f'action [{action}] received')
        assert action < self.action_space_dim, 'invalid action value'

        is_stop = action == self.action_space_dim - 1
        is_reinit = action == self.action_space_dim - 2

        if is_stop or self.time_step + 1 >= self.horizon:
            final_reward = self._compute_final_reward()
            self._append_log(np.zeros(self.K), self._get_phi_val(), self._get_verb_action(action), final_reward)
            self.log(f'received STOP signal (or exceeded horizon), final reward is: [{final_reward}]')
            return None, final_reward, True, {}

        elif is_reinit:
            self.log(f'received RE-INIT signal')

            self._init_backbone()
            self.ll = StateLinkedList(savedir=self.savedir, dim=self.K + 2)

        # lr scaling; if 0 just train
        elif action == 1:
            self._scale_lr(0.8)
        elif action == 2:
            self._scale_lr(0.2)
        elif action == 3:
            self._scale_lr(1.2)
        elif action == 4:
            self._scale_lr(1.8)

        # do training
        loss_vec = self._train_one_cycle()

        self._add_observation(loss_vec, self._get_phi_val())

        # get last H observations
        o_history = self.ll.get_observations(self.H)

        # compute intermediate reward
        step_reward = self._compute_intermediate_reward()
        self.log(f'reward at the end of time step is [{step_reward}]')

        self._append_log(loss_vec, self._get_phi_val(), self._get_verb_action(action), step_reward)
        return o_history, step_reward, False, {}

    def _scale_lr(self, scale_factor):

        for g in self.opt.param_groups:
            g['lr'] *= scale_factor

        self._curr_lr *= scale_factor

    def _train_one_cycle(self, loss_vec=None, steps=0):
        if loss_vec is None:
            self.log('training loop started ...')
            loss_vec = np.zeros(self.K)

        self.backbone.train()
        for i, batch in enumerate(self.trndl):
            inputs, labels = batch[0], batch[1]

            if self.device:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.opt.zero_grad()

            outputs = self.backbone(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.opt.step()

            steps += 1

            if steps >= self.T:
                self.time_step += 1  #  this defines the time step
                self.log('training loop done!')
                return loss_vec

            if steps % self.sampling_interval == 0:
                loss_vec[(steps // self.sampling_interval) - 1] = loss.item()

        return self._train_one_cycle(loss_vec=loss_vec, steps=steps)

    def _compute_final_reward(self):
        return self._get_phi_val() * self._fr_scale

    def _compute_intermediate_reward(self):  # check logic here
        delta = self._get_phi_val() - self._prev_phi_val
        if delta > 0:
            return self._inter_reward
        else:
            return -self._inter_reward

    def reset(self):

        return self.init(self.backbone, self.phi, self.savedir, self.trnds, self.valds,
                         T=self.T, H=self.H, S=self.sampling_interval, lr_init=self.lr_init,
                         inter_reward=self._inter_reward, final_reward_scale=self._fr_scale,
                         horizon=self.horizon, criterion=self.criterion,
                         num_workers=self.num_workers, bs=self.bs, v=self.v, device=self.device)

    def render(self, mode='human', close=False):
        """render observation; for that need to add logs"""
        pass

    def plot_loss(self):
        history = np.concatenate(list(self.logloss.values()), axis=0)
        ax = sns.lineplot(x=range(len(history)), y=history)
        ax.set(xlabel='batch updates (v. line is step delim.)', ylabel='loss value')

        for i in range(1, len(self.logloss)):
            plt.axvline(self.K * i, ls='-')

        plt.title('loss plot vs batch update')

    def plot_mdp(self, figsize=(7, 7)):
        """plot reward and phi value"""
        f, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
        sns.despine(left=True)
        # 'phi', 'reward', 'action', 'weights history'

        sns.lineplot(data=self.logmdp, x='t', y='phi', ax=axes[0, 0])
        axes[0, 0].set_title('phi val')

        sns.lineplot(data=self.logmdp, x='t', y='reward', ax=axes[0, 1])
        axes[0, 1].set_title('reward')

        cum_reward_mdp = self.logmdp.cumsum(axis='reward')

        sns.lineplot(data=cum_reward_mdp, x='t', y='reward', ax=axes[1, 0])
        axes[1, 0].set_title('cumulative reward')

        sns.lineplot(data=self.logmdp, x='t', y='weights history', ax=axes[1, 1])
        axes[1, 1].set_title('weights history')


class ObservationAndState:
    def __init__(self, param_dict: dict, o: np.array):
        self.param_dict = param_dict
        self.o = o

        self.dim = self.o.size

    def to_dict(self):
        return {
            'param_dict': self.param_dict,
            'o': self.o
        }

    def __repr__(self):
        return f"ObservationAndState Object --> o={self.o} param_dict={self.param_dict}"


class StateLinkedList:

    def __init__(self, savedir: Path, dim: int):

        if type(savedir) == str:
            savedir = Path(savedir)

        assert savedir.exists() and savedir.is_dir(), "please make sure save path exists and is directory"
        assert dim > 0

        self.savedir = savedir
        self.dim = dim  #  observation dimension

        self.len = 0  # the id of the next node

    def get_observations(self, size):
        os = []
        if size > self.len:
            for _ in range(size - self.len):
                os.append(np.zeros(self.dim))
            size = self.len

        os += [self[i].o for i in range(size)]
        #         print('o dim: ', self.dim)
        #         print('os shape: ', [o.shape for o in os])
        #         print('os: ', os)
        return np.vstack(os)

    def append(self, state: ObservationAndState):

        new_node_path = self.node_path(self.len)
        torch.save(state, new_node_path)
        self.len += 1

    def node_path(self, idx) -> Path:
        return self.savedir / f'state_{idx}.ckpt'

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        if idx >= self.len:
            raise ValueError('idx too large')

        return torch.load(self.node_path(idx))

    def rewind(self, steps: int):

        steps = min(steps, self.len)

        for i in range(1, steps + 1):
            nodepath = self.node_path(self.len - i)
            nodepath.unlink()

        self.len -= steps
