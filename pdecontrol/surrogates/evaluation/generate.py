import gym
import json
import argparse

import torch
import numpy as np
from torch.utils.data import TensorDataset

import pdegym


parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--config", type=str, default="{}")
args = parser.parse_args()


if __name__ == "__main__":
    # Load environment configuration passed as JSON dictionary.
    config = json.loads(args.config)
    env = gym.make(args.env, config=config, new_step_api=True)

    data = []

    for episode in range(args.episodes):
        obs = env.reset()
        terminated, truncated, episode = False, False, []

        while not terminated and not truncated:
            action = env.action_space.sample()
            nxt, rew, terminated, truncated, info = env.step(action)

            episode.append((obs, action, nxt, rew, terminated, truncated))
            obs = nxt

        data.append(episode)

    # Stack samples to (obs, actions, rewards, nxt, dones) tensor.
    data = [tuple(zip(*ep)) for ep in data]
    obs, actions, nxt, rewards, terminated, truncated = zip(*data)

    obs = np.array(obs, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    nxt = np.array(nxt, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    terminated = np.array(terminated, dtype=np.bool8)
    truncated = np.array(truncated, dtype=np.bool8)
    steps = np.arange(obs.shape[1], dtype=np.int32).reshape(1, -1)
    steps = np.repeat(steps, args.episodes, axis=0)

    obs = torch.FloatTensor(obs)
    actions = torch.FloatTensor(actions)
    nxt = torch.FloatTensor(nxt)
    rewards = torch.FloatTensor(rewards)
    terminated = torch.BoolTensor(terminated)
    truncated = torch.BoolTensor(truncated)
    steps = torch.LongTensor(steps)

    # Save TensorDataset at output location.
    data = TensorDataset(obs, actions, nxt, rewards, terminated, truncated, steps)
    torch.save(data, args.output)
