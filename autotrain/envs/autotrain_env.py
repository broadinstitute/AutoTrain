from collections import namedtuple

import gym
from gym import spaces

from autotrain.envs.thresholdout import Thresholdout
import autotrain.envs.utils as utils

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

"""
TODO:
    - ATE
        - step mechanics
            - observation
                - each channel for image
        
    -Clf
        - scaling the batch size
            - custom data loaders


"""

Clf = namedtuple('Clf', ['history', 'result'])


class ClfEngine:
    def __init__(self, model, trnds, valds, phi: callable,
                 criterion: callable = nn.CrossEntropyLoss(), opt: callable = optim.SGD,
                 lr_init=3e-4, bs_init=16, dev=None, v=False):

        self.history = [[], [], []]  # loss, lr, bs

        self.dev = dev
        self.v = v

        self.lr_init = self._curr_lr = lr_init
        self.bs_init = self._curr_bs = bs_init

        self.model = model
        self.opt_cls = opt
        self.opt = self.opt_cls(self.model.parameters(), lr=self._curr_lr)
        self.criterion = criterion

        # DATA

        self.trnds = trnds
        self.valds = valds

        self.trndl, self.valdl = utils.create_dls(self.trnds, self.valds)
        # set bs to self._curr_bs

        # Thresholdout & statistic of interest
        self.thresholdout = Thresholdout(self.trndl, self.valdl)
        self.phi = phi
        self._phi_func = partial(self.phi, model=self.model)  #   does partial work on this

        self.glob_step = 0

    # MAIN API

    def reinit(self):
        # clear history
        self.init_model()
        pass

    def init_model(self):

        utils.init_params(self.model)

        if self.dev:
            self.model.to(self.dev)

        self.opt = self.opt_cls(self.model.parameters(), lr=self._curr_lr)

    def scale_bs(self, scale_factor):
        # TODO
        pass

    def scale_lr(self, scale_factor):

        for g in self.opt.param_groups:
            g['lr'] *= scale_factor

        self._curr_lr *= scale_factor

    def do_updates(self, N: int):
        # data should not repeat

        step = 0

        self.model.train()
        for i, batch in enumerate(self.trndl):
            inputs, labels = batch[0], batch[1]

            if self.dev:
                inputs, labels = inputs.to(self.dev), labels.to(self.dev)

            self.opt.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()

            self.opt.step()
            step += 1

            if step >= N:
                self.log('training loop done!')
                self._append_history(loss.item())
                return

    # HELPER
    def save(self, p: Path):
        assert p.is_dir()

        torch.save(self.model.state_dict(), p / 'model.ckpt')
        utils.pkl_save(self.history, p / 'history.pkl')

    def _append_history(self, loss: float):
        self.history[0] += [loss]
        self.history[1] += [self._curr_lr]
        self.history[2] += [self._curr_bs]

    def _get_phi(self):
        self.log('initialised phi value: started ...')
        phi_val = self.thresholdout.verify(self._phi_func)
        self.log('initialised phi value: done')
        return phi_val

    def toclf(self) -> Clf:
        h = np.array(self.history)
        result = self.thresholdout.verify(self._phi_func)
        return Clf(h, result)

    def log(self, s):
        if self.v: print('[clf_enigine] ', s)


class AutoTrainEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def set_baseline(self, history, result):
        """
        method to be used when changing the baseline model/performance
        critical for slef play and final reward fomulation
        history:
            - Nx3 sized matrix where N is the number of batch updates
                - 0th dim is loss
                - 1th dim is lr used
                - 2nd dim is bs used
            -
        result:
            - scalar
                - phi value

        """

        self._baseline = Clf(history, result)

    # pass in just two Clf and ClfEngine?
    def init(self, baseline: Clf, competitor: ClfEngine, savedir: Path,
             T=30, horizon=50, step_reward=0.1, terminal_reward=10, update_penalty=0.1,
             num_workers=4, v=False, device=None):

        self.reward_range = None
        self.action_space = spaces.Box(low=np.array([0., 0., 0., 0.]), high=np.array([10., 10., 1., 1.]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3, 128, 128),
                                            dtype=np.float32)  # TODO image low high

        # experiment parameter setup

        self.T = T  # how many batch updates in one time step
        self.horizon = horizon  # time steps i.e. multiple of T

        self.step_reward = step_reward
        self.terminal_reward = terminal_reward
        self.update_penalty = update_penalty

        self.v = v  # is verbose
        self.device = device

        self.num_workers = num_workers
        self.savedir = savedir

        # clf and baseline packaging

        self._baseline = baseline
        self._competitor = competitor

        # data collection

        self.logmdp = pd.DataFrame(columns=['t', 'phi', 'reward', 'action', 'clf. optim. steps', 'lr'])  #  length of ll

        self.logloss = dict()  # array of loss vectors; idx is timestep

        # self._append_log(self._get_phi_val(), "ENV INIT", 0)
        self.time_step = 0

        self.log(f'environment initialised : {self.__repr__()}')

    def _append_log(self, loss_vec: np.array, phival: float, action_vrb: str, reward: float):

        # vrb actions are good, easier to inspect

        self.logmdp.loc[len(self.logmdp)] = [self.time_step, phival, reward, action_vrb, self.ll.len, self._curr_lr]

        self.logloss[self.time_step] = loss_vec

    def save(self, savedir: Path):

        assert savedir.is_dir() and savedir.exists()

        self.logmdp.to_csv(savedir / 'env_mdp_log.csv')

        with (savedir / 'env_loss.pkl').open('wb') as fp:
            pkl.dump(self.logloss, fp)

        with (savedir / 'env_params.pkl').open('wb') as fp:
            pkl.dump(self.get_env_params(), fp)

    def seed(self, seed=None):
        torch.seed
        np.random.seed
        pass

    def step(self, action: np.ndarray):
        """
        step(self, action: int):
            @action: scaling params 4-dim; by index:
                - 0: lr scale
                - 1: bs scale
                - 2: re init prob
                - 3: stop prob

        """

        self.log(f'action [{action}] recieved')
        assert self.action_space.contains(action), 'invalid action provided'

        is_stop = torch.rand() > action[-1]
        is_reinit = torch.rand() > action[-2]

        if is_stop or self.time_step + 1 >= self.horizon:
            final_reward = self._compute_final_reward()
            self.log(f'received STOP signal (or exceeded horizon), final reward is: [{final_reward}]')
            return None, final_reward, True, {}

        if is_reinit:
            self._competitor.reinit()

        # apply changes
        self._competitor.scale_bs(action[0])
        self._competitor.scale_bs(action[1])

        self._competitor.do_updates(self.T)

        self.time_step += 1

        step_reward = self._compute_step_reward()
        self.log(f'reward at the end of time step is [{step_reward}]')

        return self._make_o(), step_reward, False, {}

    def _make_o(self) -> np.array:
        # plotting: set xlim with horizon

        # convey motion?
        # the agent should be able to reason from a stationary plot

        # use those two up to a time step

        self._competitor.history
        self._baseline.history
        #  use each channel to convey different thing like, channel 0 for competitor loss

        pass

    def _compute_final_reward(self) -> float:

        r = self.terminal_reward

        competitor = self._competitor.toclf()

        if self._baseline.result > competitor.result:
            r *= -1

        r += -self.update_penalty * len(competitor.history)

        return r

    def _compute_step_reward(self) -> float:
        return -self.step_reward

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        # TODO
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
