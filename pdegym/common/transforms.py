from typing import List, Sequence, Iterable
from abc import abstractmethod

import torch
import numpy as np

from pdecontrol.mbrl.types import Sample


class Transform:
    @abstractmethod
    def __call__(self, values):
        pass

    def update(self, values):
        pass

    class _Inverse:
        def __init__(self, transf: 'Transform'):
            self.transf = transf

        @abstractmethod
        def __call__(self):
            pass

        def update(self, values):
            values = self(values)
            self.transf.update(values)

        @property
        def Inverse(self):
            return self.transf

    @property
    def Inverse(self):
        return self._Inverse(self)

    @staticmethod
    def convert(values):
        if isinstance(values, np.ndarray):
            return torch.from_numpy(values), np.ndarray

        return values, torch.Tensor

    @staticmethod
    def unconvert(values, vtype):
        if vtype == np.ndarray:
            values = values.detach().numpy()

        return values


class Identity(Transform):
    def __call__(self, values):
        return values

    class _Inverse(Transform._Inverse):
        def __call__(self, values):
            return values


class Normalize(Transform):
    def __init__(self, aggregate=False, batched=False, frozen=False, epsilon=1e-4):
        super().__init__()
        self.aggregate = aggregate
        self.batched = batched
        self.frozen = frozen

        self.mean, self.var = None, None

        if self.aggregate and self.batched:
            self.dim = (0, 1, 2,)
        elif self.aggregate and not self.batched:
            self.dim = (0, 1)
        elif not self.aggregate and self.batched:
            self.dim = (0, 1)
        else:
            self.dim = (0,)

        # self.dim = (1,) if self.aggregate else dim
        self.epsilon = epsilon
        self.count = 0

    def reset(self):
        self.mean, self.var = None, None
        self.count = 0

    def __call__(self, values):
        values, vtype = Transform.convert(values)

        values = (values - self.mean) / torch.sqrt(self.var + self.epsilon)

        return Transform.unconvert(values, vtype=vtype)

    def update(self, values):
        if self.frozen:
            return

        values, vtype = Transform.convert(values)

        bsize, _, _ = values.shape

        if self.mean is None:
            self.mean = torch.zeros_like(values, dtype=torch.float32)
            self.mean = torch.sum(self.mean, dim=self.dim, keepdim=True)

        if self.var is None:
            self.var = torch.zeros_like(values, dtype=torch.float32)
            self.var = torch.sum(self.var, dim=self.dim, keepdim=True)

        # In case `aggregate` is `0`, the mean and std. are computed for each dimension.
        # If `aggregate` is None, the mean and std. are aggregated over e.g. the entire pressure field.
        batch_mean = torch.mean(
            values, dim=self.dim, keepdim=True, dtype=torch.float32
        )
        batch_var = torch.var(values, dim=self.dim, keepdim=True)

        delta = batch_mean - self.mean
        tot_count = self.count + bsize

        # taken from https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py
        self.mean = self.mean + delta * bsize / tot_count
        m_a = self.var * self.count
        m_b = batch_var * bsize
        M2 = m_a + m_b + np.square(delta) * self.count * bsize / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    class _Inverse(Transform._Inverse):
        def __call__(self, values):
            values, vtype = Transform.convert(values)

            values = (
                values * torch.sqrt(self.transf.var + self.transf.epsilon)
                + self.transf.mean
            )

            return Transform.unconvert(values, vtype=vtype)


class ScaleTransform(Transform):
    def __init__(
        self, scale=(-1.0, 1.0), bounds=(-np.inf, np.inf), aggregate=False, batched=False, frozen=False
    ):
        super().__init__()

        self.aggregate = aggregate
        self.batched = batched
        self.frozen = frozen

        if self.aggregate and self.batched:
            self.dim = (0, 1, 2,)
        elif self.aggregate and not self.batched:
            self.dim = (0, 1)
        elif not self.aggregate and self.batched:
            self.dim = (0, 1)
        else:
            self.dim = (0,)

        self.lower, self.upper = scale
        self.lower = torch.from_numpy(np.asarray(self.lower, dtype=np.float32))
        self.upper = torch.from_numpy(np.asarray(self.upper, dtype=np.float32))

        self.vmin, self.vmax = bounds
        self.vmin = torch.from_numpy(np.asarray(self.vmin, dtype=np.float32))
        self.vmax = torch.from_numpy(np.asarray(self.vmax, dtype=np.float32))

        if self.aggregate and self.vmin.ndim > 1 and self.vmax.ndim > 1:
            self.vmin = torch.amin(self.vmin, dim=self.dim, keepdim=True)
            self.vmax = torch.amax(self.vmax, dim=self.dim, keepdim=True)

    def __call__(self, values):
        values, vtype = Transform.convert(values)
        scaled = (values - self.vmin) / (self.vmax - self.vmin) * (
            self.upper - self.lower
        ) + self.lower

        return Transform.unconvert(scaled, vtype=vtype)

    def update(self, values):
        if self.frozen:
            return

        values, vtype = Transform.convert(values)

        if np.isneginf(self.vmin):
            self.vmin = torch.zeros_like(values, dtype=torch.float32)
            self.vmin = torch.mean(self.vmin, dim=self.dim, keepdim=True)
            self.vmin = np.inf * (1.0 + self.vmin)

        if np.isposinf(self.vmax):
            self.vmax = torch.zeros_like(values, dtype=torch.float32)
            self.vmax = torch.mean(self.vmax, dim=self.dim, keepdim=True)
            self.vmax = -1.0 * np.inf * (1.0 + self.vmax)

        vmin = torch.amin(values, dim=self.dim, keepdim=True)
        self.vmin = torch.minimum(vmin, self.vmin)

        vmax = torch.amax(values, dim=self.dim, keepdim=True)
        self.vmax = torch.maximum(vmax, self.vmax)

    class _Inverse(Transform._Inverse):
        def __call__(self, values):
            values, vtype = Transform.convert(values)

            rescaled = (values - self.transf.lower) / (
                self.transf.upper - self.transf.lower
            ) * (self.transf.vmax - self.transf.vmin) + self.transf.vmin

            return Transform.unconvert(rescaled, vtype=vtype)


class FuncTransform(Transform):
    def __init__(self, transf, inverse=None):
        super().__init__()
        self.transf = transf
        self.inverse = inverse

    def __call__(self, *args):
        args, vtypes = zip(*map(Transform.convert, args))
        result = self.transf(*args)
        return Transform.unconvert(result, vtype=vtypes[0])

    class _Inverse(Transform._Inverse):
        def __call__(self, *args):
            args, vtypes = zip(*map(Transform.convert, args))
            result = self.transf.inverse(*args)
            return Transform.unconvert(result, vtype=vtypes[0])


class SensorTransform(Transform):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def __call__(self, values):
        values, vtype = Transform.convert(values)
        values = values[..., int(self.stride / 2) :: self.stride]
        return Transform.unconvert(values, vtype=vtype)

    class _Inverse(Transform._Inverse):
        def __call__(self, values):
            values, vtype = Transform.convert(values)
            if self.transf.stride > 1:
                raise NotImplementedError()

            return Transform.unconvert(values, vtype=vtype)


class GaussianForcing(Transform):
    def __init__(self, x: Sequence, Xi: Sequence, sigma: float, L:float, N:int):
        self.sigma, self.L, self.N = sigma, L, N
        self.x, self.Xi = torch.as_tensor(x), torch.as_tensor(Xi)
        self.x, self.Xi = self.x.to(dtype=torch.float32), self.Xi.to(dtype=torch.float32)

        self.xi = (self.L * self.Xi).reshape(-1, 1)

        # Define Gaussian forcing matrix.
        self.forcing = torch.exp((-((self.x - self.xi) ** 2.0) / (2.0 * sigma**2)))
        self.forcing = self.forcing / np.sqrt(2.0 * np.pi * self.sigma)

    def __call__(self, values):
        values, vtype = Transform.convert(values)
        values = values @ self.forcing
        return Transform.unconvert(values, vtype=vtype)

    class _Inverse(Transform._Inverse):
        def __init__(self, transf: Transform):
            super().__init__(transf)

            # Define inverse of forcing pattern, i.e., maps pattern to action.
            self.xpos = self.transf.Xi.reshape(-1, 1)
            self.xpos = (self.transf.N * self.xpos).to(dtype=torch.long).reshape(-1)
            self.inv_forcing = torch.inverse(self.transf.forcing[:, self.xpos])

        def __call__(self, values):
            values, vtype = Transform.convert(values)
            values = values[..., self.xpos] @ self.inv_forcing
            return Transform.unconvert(values, vtype=vtype)


class BatchTransform(Transform):
    def __init__(self, transform: Transform):
        super().__init__()
        self.transform = transform

    def __call__(self, values):
        values, vtype = Transform.convert(values)
        values = [self.transform(value) for value in values]
        values = torch.stack(values, dim=0)
        return Transform.unconvert(values, vtype=vtype)

    def update(self, values):
        values, vtype = Transform.convert(values)
        for value in values:
            self.transform.update(value)

    class _Inverse(Transform._Inverse):
        def __init__(self, transf: Transform):
            super().__init__(transf)
            self.transform = self.transf.transform.Inverse

        def __call__(self, values):
            values, vtype = Transform.convert(values)
            values = [self.transform(value) for value in values]
            values = torch.stack(values, dim=0)
            return Transform.unconvert(values, vtype=vtype)


class Operation(Transform):
    def __init__(self, transforms: List[Transform]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, values):
        values, vtype = Transform.convert(values)
        for transf in self.transforms:
            values = transf(values)

        return Transform.unconvert(values, vtype=vtype)

    def update(self, values):
        values, vtype = Transform.convert(values)
        for transf in self.transforms:
            transf.update(values)
            values = transf(values)

        return values

    class _Inverse(Transform._Inverse):
        def __init__(self, transf: Transform):
            super().__init__(transf)
            self.transfs = [transf.Inverse for transf in self.transf.transforms]
            self.transfs = list(reversed(self.transfs))

        def __call__(self, values):
            values, vtype = Transform.convert(values)
            for transf in self.transfs:
                values = transf(values)

            return Transform.unconvert(values, vtype=vtype)


class SampleTransform(Transform):
    def __init__(self, otransf: BatchTransform = None, atransf: BatchTransform = None):
        super().__init__()

        self.otransf, self.atransf = otransf, atransf

        if self.otransf is None:
            self.otransf = BatchTransform(Identity())

        if self.atransf is None:
            self.atransf = BatchTransform(Identity())

        self.otransf = list(self.otransf) if isinstance(self.otransf, Iterable) else [self.otransf]
        self.otransf = Operation(self.otransf)
        self.atransf = list(self.atransf) if isinstance(self.atransf, Iterable) else [self.atransf]
        self.atransf = Operation(self.atransf)

    def __call__(self, sample: Sample):
        obs, actions, nxtobs, rewards, terminated, truncated, steps = sample

        obs = self.otransf(obs)
        nxtobs = self.otransf(nxtobs)
        actions = self.atransf(actions)

        return Sample(obs, actions, nxtobs, rewards, terminated, truncated, steps)

    @property
    def Inverse(self):
        return SampleTransform(
            otransf=self.otransf.Inverse, atransf=self.atransf.Inverse
        )
