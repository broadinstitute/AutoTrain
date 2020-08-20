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
from PIL import Image

from pathlib import Path
from functools import partial
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

import io
from tqdm.notebook import tqdm

Clf = namedtuple('Clf', ['history', 'result'])  # 3; loss, lr, bs


class ClfEngine:
    def __init__(self, model: nn.Module, trnds: torchdata.Dataset, valds: torchdata.Dataset, phi: callable,
                 criterion: callable = nn.CrossEntropyLoss(), opt: callable = optim.SGD,
                 lr_init=3e-4, bs_init=16, max_lr=3, dev=None, v=False):

        self.history = [[], [], []]  # loss, lr, bs
        self.result = []
        self.optim_step = 0  # num batch updates

        self.dev = dev
        self.v = v

        # need to set max bs and max lr
        self.lr_init = self._curr_lr = lr_init
        self.bs_init = self._curr_bs = bs_init

        # DATA

        self.trnds = trnds
        self.valds = valds

        self.trndl, _, self.valdl = utils.create_dls(self.trnds, self.valds, bs=self._curr_bs)

        self._max_bs = len(trnds) / 10
        self._max_lr = max_lr

        self.model = model.to(self.dev) if self.dev else model
        self.opt_cls = opt
        self.opt = self.opt_cls(self.model.parameters(), lr=self._curr_lr)
        self.criterion = criterion

        # Thresholdout & statistic of interest
        self.thresholdout = Thresholdout(self.trndl, self.valdl)
        self.phi = phi
        self._phi_func = partial(self.phi, model=self.model)  #   does partial work on this

    # MAIN API

    def reinit(self):
        self.init_model()
        self.history = [[], [], []]
        self.optim_step = 0
        self.log("re-init complete")

    def init_model(self):

        utils.init_params(self.model)

        if self.dev:
            self.model.to(self.dev)

        self.opt = self.opt_cls(self.model.parameters(), lr=self._curr_lr)

    def scale_bs(self, scale_factor):
        # originally wanted to do custom data loaders so that data wouldn't repeat
        # but as num. updates << len. dataset we  will just get by with  reinitialising
        self._curr_bs = min(self._max_bs, int(self._curr_bs * scale_factor))
        self.trndl, _, self.valdl = utils.create_dls(self.trnds, self.valds, bs=self._curr_bs)

        self.log(f"scaled BS by [{scale_factor}]; BS=[{self._curr_bs}]")

    def scale_lr(self, scale_factor):

        new_lr = min(self._max_lr, self._curr_lr * scale_factor)
        self.log(f"scaled LR by [{scale_factor}]; LR=[{self._curr_lr}]")

        for g in self.opt.param_groups:
            g['lr'] = new_lr

        self._curr_lr = new_lr

    def do_updates(self, N: int):

        if not N:
            self.log('training loop: done!')
            self.optim_step = len(self.history[0])
            return

        self.log(f'training loop: started for [{N}] updates; BS=[{self._curr_bs}] LR=[{self._curr_lr}]!')

        self.model.train()

        for i, batch in tqdm(enumerate(self.trndl), total=len(self.trndl)):
            if N <= 0:  # to exhaust the data-loader; otherwise causes child process errors
                continue

            inputs, labels = batch[0], batch[1]

            if self.dev:
                inputs, labels = inputs.to(self.dev), labels.to(self.dev)

            self.opt.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.opt.step()

            self._append_history(loss.item())

            N -= 1

        self.do_updates(N)

    # HELPER
    def save(self, p: Path):
        assert p.is_dir()

        torch.save(self.model.state_dict(), p / 'model.ckpt')
        utils.pkl_save(self.history, p / 'history.pkl')

    def _append_history(self, loss: float):
        self.history[0] += [loss]
        self.history[1] += [self._curr_lr]
        self.history[2] += [self._curr_bs]

    def test(self) -> float:
        return self.thresholdout.verify(self._phi_func)

    def clf(self) -> Clf:
        h = np.array(self.history)
        result = np.array(self.result)
        return Clf(h, result)

    def log(self, s):
        if self.v: print('[clf_engine] ', s)


class AutoTrainEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def init(self, baseline: Clf, competitor: ClfEngine, savedir: Path,
             U: int = 30, horizon: int = 50, step_reward: float = 0.1, terminal_reward: float = 10,
             update_penalty: float = 0.1, test_interval: int = 10, o_dim=(600,900),
             num_workers=4, v=False, device=None):

        self.reward_range = None
        self.action_space = spaces.Box(low=np.array([0., 0., 0., 0.]), high=np.array([10., 10., 1., 1.]),
                                       dtype=np.float32)
        # 7 channels:  loss, lr, bs for competitor and baseline + one for results monitoring

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(7, *o_dim),
                                            dtype=np.float32)  # TODO image low high

        # experiment parameter setup

        self.U = U  # how many batch updates in one time step
        self.horizon = horizon  # time steps i.e. multiple of T

        self.step_reward = step_reward
        self.terminal_reward = terminal_reward
        self.update_penalty = update_penalty

        self.test_interval = test_interval  # how often to evaluate the performance of the competitor

        self.v = v  # is verbose
        self.device = device

        self.num_workers = num_workers
        self.savedir = savedir

        # clf and baseline packaging

        self._baseline = baseline
        self._competitor = competitor

        # data collection

        self.logmdp = pd.DataFrame(
            columns=['t', 'reward', 'optim. steps', 'lr.scale', 'lr.val', 'bs.scale', 'bs.val',
                     'p(re-init)', 'p(stop)'])

        self.time_step = 0

        self.log(f'environment initialised : {self.__repr__()}')

    def log_step(self, reward: float, action: np.array):
        lr_scale, bs_scale, p_reinit, p_stop = action
        lr_val, bs_val = self._competitor._curr_lr, self._competitor._curr_bs
        optim_step = self._competitor.optim_step

        self.logmdp.loc[len(self.logmdp)] = [self.time_step, reward, optim_step, lr_scale, lr_val, bs_scale, bs_val,
                                             p_reinit, p_stop]

    def save(self, savedir: Path):

        assert savedir.is_dir() and savedir.exists()

        self.logmdp.to_csv(savedir / 'mdplog.csv')

    @staticmethod
    def seed(seed=2020):
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def set_config(size=(9, 9)):
        sns.set(rc={'figure.figsize': size,
                    'axes.facecolor': 'black',
                    'figure.facecolor': 'black'})

    @staticmethod
    def reset_config():
        sns.axes_style("darkgrid")

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

        is_stop = np.random.rand() < action[-1] # TODO: thresholding exp.
        is_reinit = np.random.rand() < action[-2]

        if is_stop or self.time_step + 1 >= self.horizon:
            final_reward = self._compute_final_reward()
            self.log(f'received STOP signal (or exceeded horizon), final reward is: [{final_reward}]')
            return None, final_reward, True, self._competitor.clf()

        if is_reinit:
            self._competitor.reinit()

        # apply changes
        self._competitor.scale_bs(action[0])
        self._competitor.scale_lr(action[1])

        self._competitor.do_updates(self.U)

        self.time_step += 1
        if self.time_step % self.test_interval == 0:
            self._competitor.result += [self._competitor.test()]

        step_reward = self._compute_step_reward()
        self.log(f'reward at the end of time step is [{step_reward}]')

        O, debug = self._make_o()
        return O, step_reward, False, dict(plots=debug)

    def _make_plot(self, data, color='b', ax=None):
        plt.axis('off')
        x = range(len(data))
        ax = sns.lineplot(y=data, x=x, color=color, ax=ax)
        ax.set_xlim(0, self.U * self.horizon)  #  possibly need to plt.close(fg)

        # ax.set_ylim(0, self._competitor._max_lr)
        return ax

    def _plot_to_vec(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        im = Image.open(buf).convert("L")
        im.thumbnail(self.observation_space.shape[1:], Image.ANTIALIAS)
        return np.asarray(im), im

    def _make_o(self) -> np.array:
        # plotting: set xlim with horizon

        # convey motion?
        #  use each channel to convey different thing like, channel 0 for competitor loss
        # the agent should be able to reason from a stationary plot
        # TODO NOTE: maybe be good to convolute spatial i,j coordinates too?

        O = np.zeros(self.observation_space.shape)
        d = 0
        plts = []

        for player in [self._baseline, self._competitor]:
            for i in range(3):
                data = player.history[i]

                if len(data):
                    ax = self._make_plot(data)
                    vec, im = self._plot_to_vec(ax.figure)
                    plts += [im]
                else:
                    vec = 0

                O[d, ...] = vec
                d += 1

        target = self._baseline.result[-1]

        r_ax = self._make_plot([target] * (self.U * self.horizon))
        r_ax = sns.scatterplot(x=range(len(self._competitor.result)), y=self._competitor.result, ax=r_ax, marker="+")
        O[-1, ...], im = self._plot_to_vec(r_ax.figure)

        plts += [im]

        return O, plts  #  plts for debug

    def _compute_final_reward(self) -> float:

        r = self.terminal_reward

        competitor = self._competitor.clf()

        if self._baseline.result > competitor.result:
            r *= -1

        r -= self.update_penalty * len(competitor.history)

        return r

    def _compute_step_reward(self) -> float:
        # TODO prop to delta loss between baseline vs competitor
        return -self.step_reward

    def set_baseline(self, baseline: Clf):
        self._baseline = baseline

    def reset(self):
        self._competitor.reinit()
        return self.init(baseline=self._baseline, competitor=self._competitor, savedir=self.savedir,
                         U=self.U, horizon=self.horizon, step_reward=self.step_reward,
                         terminal_reward=self.terminal_reward, update_penalty=self.update_penalty,
                         num_workers=self.num_workers, v=self.v, device=self.device)

    def render(self, mode='human', close=False):

        fg, axes = plt.subplots(2, 3)

        self._make_plot(self._competitor.history[0], ax=axes[0, 0])
        axes[0, 0].set_title('Competitor: Loss')
        axes[0, 0].set_xlabel('Batch Updates')
        axes[0, 0].set_ylabel('Loss')

        self._make_plot(self._competitor.history[1], color='g', ax=axes[0, 1])
        axes[0, 1].set_title('Competitor: LR')
        axes[0, 1].set_xlabel('Batch Updates')
        axes[0, 1].set_ylabel('LR')

        self._make_plot(self._competitor.history[2], color='r', ax=axes[0, 2])
        axes[0, 2].set_title('Competitor: BS')
        axes[0, 2].set_xlabel('Batch Updates')
        axes[0, 2].set_ylabel('BS')

        self._make_plot(self._baseline.history[0], ax=axes[1, 0])
        axes[1, 0].set_title('Baseline: Loss')
        axes[1, 0].set_xlabel('Batch Updates')
        axes[1, 0].set_ylabel('Loss')

        self._make_plot(self._baseline.history[1], color='g', ax=axes[1, 1])
        axes[1, 1].set_title('Baseline: LR')
        axes[1, 1].set_xlabel('Batch Updates')
        axes[1, 1].set_ylabel('LR')

        self._make_plot(self._baseline.history[2], color='r', ax=axes[1, 2])
        axes[1, 2].set_title('Baseline: BS')
        axes[1, 2].set_xlabel('Batch Updates')
        axes[1, 2].set_ylabel('BS')

        return fg

    def log(self, s):
        if self.v: print(f'[ATE:{self.time_step}] ', s)
