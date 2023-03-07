import torch
import numpy as np


def totorch(array):
    if not isinstance(array, torch.Tensor):
        return torch.FloatTensor(array)

    return array


def tonumpy(tensor):
    if not isinstance(tensor, np.ndarray):
        return tensor.cpu().numpy()

    return tensor


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, *args, **kwargs):
        return self.action_space.sample()


class ActionRepeatAgent:
    def __init__(self, actions):
        self.actions = actions
        assert self.actions.ndim == 4
        self.nstep = 0

    def select_action(self, *args, **kwargs):
        action = self.actions[:, self.nstep, :, :]
        self.nstep += 1
        return action
