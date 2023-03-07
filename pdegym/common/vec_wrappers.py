from copy import deepcopy
from typing import Any, Sequence, Tuple, Dict

import gym
import numpy as np

from pdegym.common.transforms import BatchTransform


class StoreNObsVecWrapper(gym.vector.VectorEnvWrapper):
    def __init__(self, env: gym.vector.VectorEnv, num_steps: int = 1) -> None:
        super().__init__(env)
        self.num_steps = num_steps

        obs = self.observation_space.sample()
        obs = np.repeat(obs[:, np.newaxis, ...], self.num_steps, axis=1)
        self.obs = np.zeros(obs.shape, dtype=self.observation_space.dtype)
        self.finals = np.zeros(obs.shape, dtype=self.observation_space.dtype)
        self.mask = np.zeros((self.env.num_envs, num_steps), dtype=np.bool8)

    def step_wait(
        self, **kwargs: Any
    ) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, rewards, terminated, truncated, infos = self.env.step_wait()

        if "final_observation" in infos:
            finals = np.asarray(list(infos["final_observation"]), dtype=np.float32)
            finals = np.expand_dims(finals, axis=1)
            self.finals[infos["_final_observation"]] = finals
            self.mask[infos["_final_observation"]] = False

        self.obs[:, 0] = obs
        self.obs = np.roll(self.obs, -1, axis=1)
        self.mask[:, 0] = True
        self.mask = np.roll(self.mask, -1, axis=1)

        return obs, rewards, terminated, truncated, infos

    def reset(self, **kwargs) -> Any:
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)

        self.obs = np.repeat(obs[:, np.newaxis, ...], self.num_steps, axis=1)
        self.mask[:, :-1] = False
        self.mask[:, -1] = True

        if kwargs.get("return_info", False):
            return obs, info
        else:
            return obs


class StoreNActionsVecWrapper(gym.vector.VectorEnvWrapper):
    def __init__(self, env: gym.vector.VectorEnv, num_steps: int = 1) -> None:
        super().__init__(env)
        self.num_steps = num_steps

        actions = self.action_space.sample()
        actions = np.repeat(actions[:, np.newaxis, ...], self.num_steps, axis=1)
        self.actions = np.zeros(actions.shape, dtype=self.observation_space.dtype)
        self.mask = np.zeros((self.env.num_envs, num_steps), dtype=np.bool8)

    def step_async(self, actions: Sequence[Any]) -> None:
        self.actions[:, 0] = actions
        self.mask[:, 0] = True

        self.actions = np.roll(self.actions, -1, axis=1)
        self.mask = np.roll(self.mask, -1, axis=1)

        return super().step_async(actions)

    def step_wait(
        self, **kwargs: Any
    ) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, rewards, terminated, truncated, infos = self.env.step_wait()

        if "final_observation" in infos:
            self.mask[infos["_final_observation"], :-1] = False

        return obs, rewards, terminated, truncated, infos

    def reset(self, **kwargs) -> Any:
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)

        self.mask[:, :] = False

        if kwargs.get("return_info", False):
            return obs, info
        else:
            return obs


class TransformActionWrapper(gym.vector.VectorEnvWrapper):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        transform: BatchTransform,
        frozen=False,
    ):
        super().__init__(env)
        self.transform = transform
        self.frozen = frozen

        low, high = self.env.action_space.low, self.env.action_space.high
        low, high = self.transform.Inverse(low), self.transform.Inverse(high)

        self.action_space = gym.spaces.Box(low, high, shape=low.shape)
        self.single_action_space = gym.spaces.Box(low[0], high[0], shape=low.shape[1:])

    def step_async(self, actions: Sequence[Any]) -> None:
        if not self.frozen:
            self.transform.update(actions)

        actions = self.transform(actions)

        return self.env.step_async(actions)

    def step_wait(
        self, **kwargs: Any
    ) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
        return self.env.step_wait(**kwargs)

    def reset(self, **kwargs) -> Any:
        return self.env.reset(**kwargs)


class TransformObsWrapper(gym.vector.VectorEnvWrapper):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        transform: BatchTransform,
        frozen=False,
    ):
        super().__init__(env)
        self.transform = transform
        self.frozen = frozen

        low, high = self.env.observation_space.low, self.env.observation_space.high
        low, high = self.transform(low), self.transform(high)
        low = np.nan_to_num(low, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        high = np.nan_to_num(high, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        self.observation_space = gym.spaces.Box(low, high, shape=low.shape)
        self.single_observation_space = gym.spaces.Box(
            low[0], high[0], shape=low.shape[1:]
        )

    def step_wait(
        self, **kwargs: Any
    ) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, rewards, terminated, truncated, infos = self.env.step_wait()

        if not self.frozen:
            self.transform.update(obs)

        obs = self.transform(obs)

        # Handle terminal observations of vector environments.
        if "final_observation" in infos:
            finals = np.asarray(list(infos["final_observation"]), dtype=np.float32)

            if not self.frozen:
                self.transform.update(finals)
                finals = self.transform(finals)
                infos["final_observation"] = finals

        return obs, rewards, terminated, truncated, infos

    def reset(self, **kwargs) -> Any:
        return_info = kwargs.get("return_info", False)
        if return_info:
            obs, info = self.env.reset(**kwargs)
            self.info = deepcopy(info)
        else:
            obs = self.env.reset(**kwargs)

        if not self.frozen:
            self.transform.update(obs)

        obs = self.transform(obs)

        if return_info:
            return obs, info

        else:
            return obs
