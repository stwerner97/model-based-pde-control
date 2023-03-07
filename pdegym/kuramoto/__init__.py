import gym
from gym.wrappers import TimeLimit, RescaleAction

from pdegym.kuramoto.kuramoto import KuramotoSivashinskyEnv
from pdegym.common.wrappers import UnFlattenActionWrapper, UnFlattenObsWrapper


def make(config={}, new_step_api=True):
    env = KuramotoSivashinskyEnv(**config)
    env = TimeLimit(env, env.unwrapped.max_episode_steps, new_step_api=new_step_api)

    return env


def make_sb3(config={}):
    env = KuramotoSivashinskyEnv(**config)

    env = UnFlattenObsWrapper(env)
    env = UnFlattenActionWrapper(env)
    env = RescaleAction(env, -1.0, 1.0)
    env = TimeLimit(env, env.unwrapped.max_episode_steps)

    return env


gym.envs.register(
    id="KuramotoSivashinskyEnv-v0",
    entry_point="pdegym.kuramoto:make",
    order_enforce=False,
    new_step_api=True,
)

gym.envs.register(
    id="KuramotoSivashinskyEnvSB3-v0",
    entry_point="pdegym.kuramoto:make_sb3",
    order_enforce=False,
)
