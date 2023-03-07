from functools import wraps
from inspect import signature, Parameter

import torch
from torch import nn


class Conv1dDerivative(nn.Module):
    def __init__(
        self, filter, resolution, kernel_size, padding=0, padding_mode="zeros"
    ):
        super().__init__()

        self.resolution = resolution

        self.filter = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )

        self.filter.weight = nn.Parameter(
            torch.FloatTensor(filter), requires_grad=False
        )

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resolution


class BatchingWrapper(nn.Module):
    def __init__(self, model:nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input):
        bsize, tsteps, channels, height = input.size()

        input = input.reshape(bsize * tsteps, channels, height)
        output = self.model(input)
        channels, height = output.size(1), output.size(2)

        return output.reshape(bsize, tsteps, channels, height)


def ignore_extra_keywords(func):
    params = signature(func).parameters.values()
    if any(p.kind == Parameter.VAR_KEYWORD for p in params):
        return func
    names = {p.name for p in params if p.kind != Parameter.VAR_POSITIONAL}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # using `names` as a closure
        return func(*args, **{k: kwargs[k] for k in (kwargs.keys() & names)})

    return wrapper
