from typing import Any, Dict, List, Sequence, Tuple

import gym
import torch
import numpy as np
from gym.vector.utils.spaces import batch_space

from pdecontrol.mbrl.types import ModelRollout
from pdecontrol.surrogates.surrogate import PDESurrogate
from pdegym.common.vec_wrappers import StoreNActionsVecWrapper, StoreNObsVecWrapper


class BaseWorldVecEnvWrapper(gym.vector.VectorEnvWrapper):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        surrogate: PDESurrogate,
        tstep: float,
        callbacks: List = None,
    ):
        super().__init__(env)

        self.surrogate = surrogate
        self.tstep = tstep
        self.callbacks = callbacks

        if self.callbacks is None:
            self.callbacks = []

        self._enabled = True

    def step_wait(
        self, **kwargs: Any
    ) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
        return self.env.step_wait(**kwargs)

    def step_async(self, actions: Sequence[Any]) -> None:
        return self.env.step_async(actions)

    def reset(self) -> Any:
        return self.env.reset()

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True
