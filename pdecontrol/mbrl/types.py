from dataclasses import dataclass
from typing import List, Union, Callable

import torch
import numpy as np
import pytorch_lightning as pl


@dataclass
class Sample:
    obs: Union[torch.FloatTensor, np.ndarray] = None
    actions: Union[torch.FloatTensor, np.ndarray] = None
    nxtobs: Union[torch.FloatTensor, np.ndarray] = None
    rewards: Union[torch.FloatTensor, np.ndarray] = None
    terminated: Union[torch.BoolTensor, np.ndarray] = None
    truncated: Union[torch.BoolTensor, np.ndarray] = None
    steps: Union[torch.IntTensor, np.ndarray] = None

    def totorch(self) -> "Sample":
        self.obs = torch.FloatTensor(self.obs)
        self.actions = torch.FloatTensor(self.actions)
        self.nxtobs = torch.FloatTensor(self.nxtobs)
        self.rewards = torch.FloatTensor(self.rewards)
        self.terminated = torch.BoolTensor(self.terminated)
        self.truncated = torch.BoolTensor(self.truncated)
        self.steps = torch.IntTensor(self.steps)

        return self

    def tonumpy(self) -> "Sample":
        self.obs = self.obs.numpy()
        self.actions = self.actions.numpy()
        self.nxtobs = self.nxtobs.numpy()
        self.rewards = self.rewards.numpy()
        self.terminated = self.terminated.numpy()
        self.truncated = self.truncated.numpy()
        self.steps = self.steps.numpy()

        return self

    def apply(self, func: Callable) -> "Sample":
        return Sample(*tuple(map(func, self)))

    def split(self, axis=0) -> List["Sample"]:
        # Swap target axis to first axis position.
        obs = np.moveaxis(self.obs, axis, 0)
        actions = np.moveaxis(self.actions, axis, 0)
        nxtobs = np.moveaxis(self.nxtobs, axis, 0)
        rewards = np.moveaxis(self.rewards, axis, 0)
        terminated = np.moveaxis(self.terminated, axis, 0)
        truncated = np.moveaxis(self.truncated, axis, 0)
        steps = np.moveaxis(self.steps, axis, 0)

        # Split sample into list of samples along first axis.
        data = (obs, actions, nxtobs, rewards, terminated, truncated, steps)
        samples = [Sample(o, a, n, r, tm, tr, st) for o, a, n, r, tm, tr, st in zip(*data)]
        
        return samples

    def __iter__(self):
        data = (
            self.obs,
            self.actions,
            self.nxtobs,
            self.rewards,
            self.terminated,
            self.truncated,
            self.steps,
        )
        return iter(data)


@dataclass
class ModelRollout:
    outputs: torch.FloatTensor = None
    inlatents: torch.FloatTensor = None
    outlatents: torch.FloatTensor = None
    deltas: torch.FloatTensor = None
    hidden: torch.FloatTensor = None

    def __iter__(self):
        return iter((self.outputs, self.inlatents, self.outlatents, self.deltas, self.hidden))


@dataclass
class PDETrainer:
    trainer: pl.Trainer = None
    early_stopping: pl.Callback = None

    def __iter__(self):
        return iter((self.trainer, self.early_stopping))