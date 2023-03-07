from typing import Any, Dict, Sequence, Tuple, Callable

import gym
import torch
import numpy as np
from torch.utils.data import Dataset, RandomSampler
from gym.vector.utils.spaces import batch_space

from pdegym.common.transforms import SampleTransform
from pdecontrol.mbrl.types import ModelRollout
from pdecontrol.surrogates.surrogate import PDESurrogate
from pdecontrol.surrogates.common.dataset import PDEDataLoader


class BaseWorldVecEnv(gym.vector.VectorEnv):
    def __init__(
        self,
        surrogate: PDESurrogate,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        max_episode_steps: int,
        stransf: SampleTransform,
        reward_func: Callable,
        num_envs: int,
        horizon: int,
        tstep: float,
    ):
        self.surrogate = surrogate
        self.max_episode_steps = max_episode_steps
        self.stransf = stransf
        self.reward_func = reward_func
        self.num_envs = num_envs
        self.horizon = horizon
        self.tstep = tstep

        low, high = observation_space.low, observation_space.high
        self.observation_space = gym.spaces.Box(low, high)

        low, high = action_space.low, action_space.high
        self.action_space = gym.spaces.Box(low, high)

    def step_async(self, actions: Sequence[Any]) -> None:
        return super().step_async(actions)

    def step_wait(
        self, **kwargs: Any
    ) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
        return super().step_wait(**kwargs)

    def reset(self) -> Any:
        return super().reset()

    def setup(self, starting: Dataset):
        # Sub-sample starting sequences. Set `num_samples` to a very high number to prevent StopIteration.
        sampler = RandomSampler(starting, replacement=True, num_samples=int(1e10))
        self.loader = iter(
            PDEDataLoader(
                starting,
                batch_size=self.num_envs,
                shuffle=False,
                sampler=sampler,
                drop_last=True,
                collate_fn=PDEDataLoader.padding_collate
            )
        )


class WorldVecEnv(BaseWorldVecEnv):
    def __init__(
        self,
        surrogate: PDESurrogate,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        max_episode_steps: int,
        stransf: SampleTransform,
        reward_func: Callable,
        num_envs: int,
        horizon: int,
        tstep: float,
    ):
        super().__init__(
            surrogate,
            observation_space,
            action_space,
            max_episode_steps,
            stransf,
            reward_func,
            num_envs,
            horizon,
            tstep
        )

        # Re-write action space of world environment.
        low, high = action_space.low, action_space.high
        low = np.squeeze(
            self.stransf.atransf.Inverse(low[np.newaxis, ...]), axis=0
        )
        high = np.squeeze(
            self.stransf.atransf.Inverse(high[np.newaxis, ...]), axis=0
        )
        self.single_action_space = gym.spaces.Box(low, high, shape=low.shape)
        self.action_space = batch_space(self.single_action_space, n=self.num_envs)

        # Re-write observation space of world environment.
        obs = observation_space.sample()
        obs = np.squeeze(
            self.stransf.otransf.Inverse(obs[np.newaxis, ...]), axis=0
        )
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs.shape)
        self.observation_space = batch_space(
            self.single_observation_space, n=self.num_envs
        )

        self.tmp = None

    def step_wait(
        self, **kwargs: Any
    ) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs = self.output.outputs.detach().cpu().numpy().squeeze(1)
        rewards = self.tmp

        # Truncate rollouts at env. time limit or when the rollout horizon is reached.
        env_limit = np.broadcast_to(
            self.timesteps >= self.max_episode_steps, (self.num_envs,)
        )
        rll_limit = np.broadcast_to(self.simulated >= self.horizon, (self.num_envs,))
        truncated = env_limit | rll_limit

        # Truncate if all sub-envs. reach the ``max_steps`` limit or ``horizon``.
        # NOTE: Some rollouts may exceed the ``max_steps``limit.
        truncated = np.broadcast_to(np.all(truncated), (self.num_envs,))

        # NOTE: Assumes that timelimits are the only stopping condition.
        terminated = np.zeros(self.num_envs, dtype=np.bool8)

        infos = {"step": self.timesteps.copy()}

        # Reset the trajectories once any sub-environment encounters a stopping condition.
        if np.any(truncated):
            infos["_final_observation"] = truncated.copy()
            infos["final_observation"] = obs[truncated]

            obs = self.reset()

        return obs, rewards, terminated, truncated, infos

    def step_async(self, actions: Sequence[Any]) -> None:
        self.surrogate.eval()
        torch.set_grad_enabled(False)

        self.simulated += 1
        self.timesteps += 1

        # Surrogate expects actions to have dimensions [B, T, C, A].
        # Passed actions have dimensions [T, C, A].
        actions = np.array(actions, dtype=np.float32)
        actions = torch.FloatTensor(actions).unsqueeze(1)

        self.output: ModelRollout = self.surrogate.rollout(
            states=self.output.outputs, actions=actions, hidden=self.output.hidden, times=0.0, targets=self.tstep
        )

        # Restore observations & actions in spaces used for the environment.
        obs = self.output.outputs.detach().squeeze(1).cpu().numpy()
        orescaled = self.stransf.otransf(obs)
        actions = actions.squeeze(1).cpu().numpy()
        arescaled = self.stransf.atransf(actions)
        
        # Estimate rewards with the function provided by the PDE gym environment.
        rewards = np.asarray([self.reward_func(o, a) for o, a in zip(orescaled, arescaled)], dtype=np.float32)
        self.tmp = torch.FloatTensor(rewards).detach().cpu().numpy()

        torch.set_grad_enabled(True)
        self.surrogate.train()

    def reset(self, **kwargs):
        self.surrogate.eval()
        torch.set_grad_enabled(False)

        # Sample starting states from dataset of gathered experience.
        sample = next(self.loader)
        states, actions, _, _, _, _, steps = sample

        times = self.tstep * torch.arange(actions.size(1))
        targets = self.tstep * actions.size(1)
        self.output: ModelRollout = self.surrogate.rollout(
            states=states, actions=actions, hidden=None, times=times, targets=targets
        )
        
        # Set steps to steps after initial warm-up phase.
        self.timesteps = steps[:, -1].numpy()
        self.simulated = 0

        obs = self.output.outputs.detach().squeeze(1).cpu().numpy()
        self.tmp = None
        
        torch.set_grad_enabled(True)
        self.surrogate.train()

        if kwargs.get("return_info", False):
            infos = {"step": self.timesteps.copy()}
            return obs, infos

        return obs
