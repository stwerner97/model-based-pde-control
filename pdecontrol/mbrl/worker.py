from typing import Callable, List, NamedTuple

import gym
import torch

from pdegym.common.vec_wrappers import StoreNObsVecWrapper, StoreNActionsVecWrapper
from pdecontrol.mbrl.types import Sample
from pdecontrol.mbrl.replay import ExperienceReplay
from pdecontrol.mbrl.callbacks import PDECallback
from pdecontrol.mbrl.world.wrappers import BaseWorldVecEnvWrapper


class PDEEnvStack(NamedTuple):
    envs: gym.vector.VectorEnv
    ostore: StoreNObsVecWrapper
    astore: StoreNActionsVecWrapper
    world_wrapper: BaseWorldVecEnvWrapper = None


class Worker:
    def __init__(
        self,
        stack: PDEEnvStack,
        callbacks: List[PDECallback] = None,
    ):
        self.stack = stack
        self.callbacks = callbacks

        if self.callbacks is None:
            self.callbacks = []

        self._last_obs = None
        self._last_stored_obs = None

    def reset(self) -> None:
        self._last_obs = None
        self._last_stored_obs = None

    def rollout(
        self,
        agent,
        stop: Callable,
        deterministic: bool = False,
    ) -> ExperienceReplay:

        replay = ExperienceReplay()

        if self._last_obs is None:
            self._last_obs = self.stack.envs.reset()
            self._last_stored_obs = self.stack.ostore.obs.copy()
            self._last_stored_obs = self._last_stored_obs[self.stack.ostore.mask]

        while not stop(replay.ntimesteps, replay.nstopped):

            with torch.no_grad():
                actions = agent.select_action(
                    self._last_obs, deterministic=deterministic
                )

            (
                self._last_obs,
                rewards,
                terminated,
                truncated,
                infos,
            ) = self.stack.envs.step(actions)

            # Copy stored obs. from intermediate storage wrapper.
            obs = self._last_stored_obs.copy()
            self._last_stored_obs = self.stack.ostore.obs.copy()
            self._last_stored_obs = self._last_stored_obs[self.stack.ostore.mask]
            nxtobs = self._last_stored_obs.copy()

            # Copy stored actions from intermediate storage wrapper.
            actions = self.stack.astore.actions.copy()
            actions = actions[self.stack.astore.mask]

            # Replace terminal obs. from stored terminal observations.
            if "final_observation" in infos:
                index = infos["_final_observation"]
                finals = self.stack.ostore.finals[index].copy()
                finals = finals[self.stack.ostore.mask[index]]
                nxtobs[index] = finals

            # Split batched sample into samples for single sub-environments.
            sample = Sample(obs, actions, nxtobs, rewards, terminated, truncated, infos["step"])
            samples = sample.split(axis=0)
            replay.add(samples)

        for callback in self.callbacks:
            callback.on_rollout_end(replay)

        return replay
