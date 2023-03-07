import gym
import numpy as np


class UnFlattenObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env: gym.Env, new_step_api=False):
        super().__init__(env, new_step_api=new_step_api)

        low = self.env.observation_space.low
        high = self.env.observation_space.high
        low, high = np.squeeze(low, axis=0), np.squeeze(high, axis=0)

        self.observation_space = gym.spaces.Box(low, high, low.shape, dtype=np.float32)

    def observation(self, obs):
        return np.expand_dims(obs, axis=0)


class UnFlattenActionWrapper(gym.core.ActionWrapper):
    def __init__(self, env, new_step_api=False):
        super().__init__(env, new_step_api=new_step_api)

        low = self.env.action_space.low
        high = self.env.action_space.high
        low, high = np.squeeze(low, axis=0), np.squeeze(high, axis=0)

        self.action_space = gym.spaces.Box(low, high, low.shape, dtype=np.float32)

    def action(self, action):
        return np.expand_dims(action, axis=0)
