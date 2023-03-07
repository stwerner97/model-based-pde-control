import importlib

import numpy as np


class Scheduler:
    def __init__(self, steptype: str):
        self.steptype = steptype

    def get_step(self, iteration, epoch, step):
        return locals().get(self.steptype)

    @staticmethod
    def factory(config):
        module = importlib.import_module("pdecontrol.surrogates.common.schedulers")
        return getattr(module, config["scheduler"])(**config)


class LinearScheduler(Scheduler):
    def __init__(
        self, steptype: str, start: int, stop: int, vmin: float, vmax: float, **kwargs
    ):
        super().__init__(steptype=steptype)
        self.start, self.stop = start, stop
        self.vmin, self.vmax = vmin, vmax

        assert self.start < self.stop

    def __call__(self, iteration=None, epoch=None, step=None):
        step = self.get_step(iteration, epoch, step)

        fraction = (step - self.start) / (self.stop - self.start)
        fraction = max(fraction, 0.0)
        value = self.vmin + fraction * (self.vmax - self.vmin)
        return np.clip(value, self.vmin, self.vmax)


class StepScheduler(Scheduler):
    def __init__(self, steptype: str, steps, values, **kwargs):
        super().__init__(steptype=steptype)
        self.steps, self.values = steps, values

    def __call__(self, iteration=None, epoch=None, step=None):
        step = self.get_step(iteration, epoch, step)

        idx = np.searchsorted(self.steps, step, side="left")
        return self.values[idx]


class FuncScheduler(Scheduler):
    def __init__(self, steptype: str, func, **kwargs):
        super().__init__(steptype=steptype)
        self.func = func

    def __call__(self, iteration=None, epoch=None, step=None):
        step = self.get_step(iteration, epoch, step)
        return self.func(step)


class ConstantLengthScheduler(Scheduler):
    def __init__(self, length:int, **kwargs):
        super().__init__(steptype="iteration")
        self.length = length

    def __call__(self, iteration=None, epoch=None, step=None):
        return self.length
