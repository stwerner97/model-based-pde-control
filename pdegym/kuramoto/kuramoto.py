import math
from typing import Dict, List

import gym
import torch
import numpy as np
from scipy.ndimage import convolve1d

from pdegym.common.transforms import FuncTransform, GaussianForcing


np.seterr(over="raise")


class KuramotoSivashinskyEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    Xi = [0, 0.25, 0.5, 0.75]  # relative positions of actuators

    eps = np.finfo(np.float32).eps
    reward_range = (-float("inf"), float("inf"))
    
    # NOTE: Flip FD coefficients (see cross-corrlation vs. convolution).
    FIRST_DERIVATIVE_SECOND_ORDER_UPWIND_FWD = [-1/4, 4/3, -3, 4, -25/12, 0, 0, 0, 0]
    FIRST_DERIVATIVE_SECOND_ORDER_UPWIND_BWD = [0, 0, 0, 0, 25/12, -4, 3, -4/3, 1/4]
    SECOND_DERIVATIVE_SIXTH_ORDER_CENTRAL = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90	]
    FOURTH_DERIVATIVE_SIXTH_ORDER_CENTRAL = [7/240, -2/5, 169/60, -122/15, 91/8, -122/15, 169/60, -2/5, 7/240]

    def __init__(
        self,
        L: float = 22.0,
        N: int = 64,
        cfg_steps: int = 250,
        Ttrans: int = 40,
        Tmax: float = 100.0,
        dt = 0.001,
        noise: float = 0.1,
        sigma: float = 0.4,
        lmbda: float = 0.0,
        objective: str = "dissipation"
    ):
        super().__init__()

        self.L = L
        self.N = N
        self.cfg_steps = cfg_steps  # number of solver steps per agent step
        self.Ttrans = Ttrans  
        self.Tmax = Tmax
        self.dt = dt
        self.noise = noise  # AWGN noise distortion of initial condition
        self.sigma = sigma  # spatial distribution of forcing.
        self.lmbda = lmbda  # coefficient of actuation penalty term
        self.objective = objective

        self.dx = self.L / self.N
        self.x = np.linspace(0.0, self.L - self.L / self.N, self.N, dtype=np.float32)
        self.max_episode_steps = math.ceil(self.Tmax / (self.dt * self.cfg_steps))

        # Define forcing pattern & inverse of forcing over spatial domain.
        self.forcing = GaussianForcing(self.x, self.Xi, self.sigma, self.L, self.N)

        self.noop = np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        def l2control(obs, *args, **kwargs):
            return (-1.0) * (1 / self.N) * torch.norm(obs) ** 2

        def dissipation(obs, phi, *args, **kwargs):
            obs, phi = np.squeeze(obs), np.squeeze(phi)
            _, (u_x, u_xx, u_xxxx) = self.rhs(obs, phi)
            return (-1) * ((u_xx * u_xx).mean() + (u_x * u_x).mean() + (obs * phi).mean())

        objective = l2control if self.objective else dissipation
        self.reward_func = FuncTransform(objective)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, len(self.Xi)), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1, self.N), dtype=np.float32)

    def step(self, action: List):
        action = np.array(action, dtype=np.float32)
        phi = np.squeeze(self.forcing(action))

        reward = 0.0
        for _ in range(self.cfg_steps):
            reward += self.reward_func(self.u, phi)
            k1, _ = self.rhs(self.u, phi)
            k2, _ = self.rhs(self.u + self.dt * k1 / 2.0, phi)
            k3, _ = self.rhs(self.u + self.dt * k2 / 2.0, phi)
            k4, _ = self.rhs(self.u + self.dt * k3, phi)

            self.u = self.u + self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        self.timestep += 1
        truncated = self.timestep >= self.max_episode_steps
        obs = self.u.reshape(1, -1)

        reward /= self.cfg_steps

        return obs, reward, False, truncated, {"step": self.timestep}

    def reset(self, seed: int = None, return_info=False, **kwargs) -> Dict:
        np.random.seed(seed)

        Tsteps = int(200.0 / self.dt / self.cfg_steps)
        self.timestep = (-1) * Tsteps

        self.u = np.random.uniform(-0.4, 0.4, size=self.N)

        for _ in range(Tsteps):
            self.step([0.0, 0.0, 0.0, 0.0])

        obs = self.u.reshape(1, -1)

        if return_info:
            return obs, {"step": self.timestep}

        return obs

    def rhs(self, u, phi):

        u_x_fwd = convolve1d(u**2, weights=self.FIRST_DERIVATIVE_SECOND_ORDER_UPWIND_FWD, mode="wrap") / self.dx
        u_x_bwd = convolve1d(u**2, weights=self.FIRST_DERIVATIVE_SECOND_ORDER_UPWIND_BWD, mode="wrap") / self.dx
        u_x = (u < 0) * u_x_fwd + (u >= 0) * u_x_bwd

        u_xx = convolve1d(u, weights=self.SECOND_DERIVATIVE_SIXTH_ORDER_CENTRAL, mode="wrap") / self.dx**2
        u_xxxx = convolve1d(u, weights=self.FOURTH_DERIVATIVE_SIXTH_ORDER_CENTRAL, mode="wrap") / self.dx**4

        rhs = -u_xxxx - u_xx - (1/2) * u_x + phi

        return rhs, (u_x, u_xx, u_xxxx)

    @property
    def time(self):
        return self.timestep * self.cfg_steps * self.dt

    @property
    def scenario(self):
        params = {
            "cfg_steps": self.cfg_steps,
            "Ttrans": self.Ttrans,
            "L": self.L,
            "N": self.N,
            "dx": self.dx,
            "Tmax": self.Tmax,
            "dt": self.dt,
            "Xi": self.Xi,
            "noise": 0.1,
            "lmbda": 1.0,
            "objective": self.objective
        }
        return params
