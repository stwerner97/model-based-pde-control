from typing import Dict, Callable

import wandb
import numpy as np

import pdecontrol.visualize as visual
from pdecontrol.mbrl.replay import ExperienceReplay


class PDECallback:
    def __init__(self, log_freq: int = 1, commit: bool = True):
        self.log_freq = log_freq
        self.commit = commit

        self.num_steps = 0
        self.num_resets = 0
        self.num_rollouts = 0

    def on_rollout_end(self, replay: ExperienceReplay):
        self.num_rollouts += 1

        if not self.num_rollouts % self.log_freq == 1:
            return

    def on_step_end(self, obs, rewards, terminated, truncated, infos):
        self.num_steps += 1

        if not self.num_steps % self.log_freq == 1:
            return

    def on_reset(self, obs, infos):
        self.num_resets += 1

        if not self.num_resets % self.log_freq == 1:
            return


class VisPDECallback(PDECallback):
    def __init__(self, plotting: Dict = {}, log_freq: int = 1, commit: bool = False):
        super().__init__(log_freq, commit)
        self.plotting = plotting

    def on_rollout_end(self, replay: ExperienceReplay):
        super().on_rollout_end(replay)

        # Sample episode from replay buffer.
        index = np.random.choice(replay.stopped)
        sample = replay.sample(index).tonumpy()

        for name, plotfnc in self.plotting.items():
            image = plotfnc(
                obs=sample.obs, actions=sample.actions, rewards=sample.rewards
            )
            wandb.log({name: [wandb.Image(image)]}, commit=self.commit)


class LogRewardDiff(PDECallback):
    def __init__(self, name, reward_func, log_freq: int, commit: bool = False):
        super().__init__(log_freq, commit)
        self.name = name
        self.reward_func = reward_func

    def on_step_end(self, obs, rewards, terminated, truncated, infos):
        super().on_step_end(obs, rewards, terminated, truncated, infos)

        rpreds = self.reward_func(obs)
        error = np.linalg.norm(rewards - rpreds, order=1)

        wandb.log({self.name: error}, commit=self.commit)


class VisRewardDiff(PDECallback):
    def __init__(
        self, name: str, reward_func: Callable, log_freq: int = 1, commit: bool = False
    ):
        super().__init__(log_freq, commit)
        self.name = name
        self.reward_func = reward_func

        self.rewards, self.rpreds = [], []

    def on_step_end(self, obs, rewards, terminated, truncated, infos):
        super().on_step_end(obs, rewards, terminated, truncated, infos)

        rpreds = self.reward_func(obs)
        self.rpreds.append(rpreds)
        self.rewards.append(rewards)

    def on_reset(self, obs, infos):
        super().on_reset(obs, infos)

        if not self.rewards or not self.rpreds:
            return

        # Select sub-environment whose rewards are plotted.
        index = np.random.randint(len(obs))

        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.rpreds = np.asarray(self.rpreds, dtype=np.float32)

        image = visual.reward_plot(
            rewards=self.rewards[:index], rpred=self.rpreds[:index]
        )
        wandb.log({self.name: [wandb.Image(image)]}, commit=self.commit)

        self.rewards, self.rpreds = [], []
