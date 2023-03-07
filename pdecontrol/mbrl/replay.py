from typing import List, Dict
from collections import defaultdict, deque

import numpy as np

from pdegym.common.transforms import SampleTransform
from pdecontrol.mbrl.types import Sample

class ExperienceReplay:
    def __init__(self, capacity: int = None):
        if capacity is None:
            capacity = np.inf

        self.capacity = capacity

        # Store sequences of e.g. observations at separate positions for each episode.
        self.obs: Dict[int, List[np.ndarray]] = defaultdict(deque)
        self.actions: Dict[int, List[np.ndarray]] = defaultdict(deque)
        self.nxtobs: Dict[int, List[np.ndarray]] = defaultdict(deque)
        self.rewards: Dict[int, List[np.ndarray]] = defaultdict(deque)
        self.terminated: Dict[int, List[np.ndarray]] = defaultdict(deque)
        self.truncated: Dict[int, List[np.ndarray]] = defaultdict(deque)
        self.steps: Dict[int, List[np.ndarray]] = defaultdict(deque)

        self.data = Sample(self.obs, self.actions, self.nxtobs, self.rewards, self.terminated, self.truncated, self.steps)

        # Store the index at which samples of each sub-environment are stored at.
        # In the first iteration, `vindex` assigns the index of each sub-environment.
        self.vindex = defaultdict(lambda: max(self.vindex.values(), default=-1) + 1)

    def extend(self, replay):

        for vid, ep in enumerate(sorted(replay.episodes)):
            vid = vid % len(replay.vindex)
            vpos = self.vindex[vid]

            self.obs[vpos].extend(replay.obs[ep].copy())
            self.actions[vpos].extend(replay.actions[ep].copy())
            self.nxtobs[vpos].extend(replay.nxtobs[ep].copy())
            self.rewards[vpos].extend(replay.rewards[ep].copy())
            self.terminated[vpos].extend(replay.terminated[ep].copy())
            self.truncated[vpos].extend(replay.truncated[ep].copy())
            self.steps[vpos].extend(replay.steps[ep].copy())

            if np.any(self.terminated[vpos]) or np.any(self.truncated[vpos]):
                self.vindex[vid] = max(self.vindex.values(), default=-1) + 1

        self.resize(self.capacity)

    def add(self, samples:List[Sample], stransf: SampleTransform=None):
        if stransf is not None:
            samples = [stransf(sample) for sample in samples]

        for vid, sample in enumerate(samples):
            vpos = self.vindex[vid]

            if sample is None:
                continue

            if stransf is not None:
                sample = stransf(sample)

            obs, actions, nxtobs, rewards, terminated, truncated, steps = sample

            self.obs[vpos].append(obs)
            self.actions[vpos].append(actions)
            self.nxtobs[vpos].append(nxtobs)
            self.rewards[vpos].append(rewards)
            self.terminated[vpos].append(terminated)
            self.truncated[vpos].append(truncated)
            self.steps[vpos].append(steps)

            if terminated or truncated:
                self.vindex[vid] = max(self.vindex.values(), default=-1) + 1

        self.resize(self.capacity)

    def sample(self, index: int = None, stransf: SampleTransform = None):

        index = np.random.choice(self.episodes) if index is None else index

        # Sample sequence of samples belonging to episode with index.
        obs = np.asarray(self.obs[index], dtype=np.float32)
        actions = np.asarray(self.actions[index], dtype=np.float32)
        nxtobs = np.asarray(self.nxtobs[index], dtype=np.float32)
        rewards = np.asarray(self.rewards[index], dtype=np.float32)
        terminated = np.asarray(self.terminated[index], dtype=np.bool8)
        truncated = np.asarray(self.truncated[index], dtype=np.bool8)
        steps = np.asarray(self.steps[index], dtype=np.int32)

        sample = Sample(obs, actions, nxtobs, rewards, terminated, truncated, steps)

        if stransf is not None:
            sample = stransf(sample)

        return sample.totorch()

    def resize(self, size):
        self.capacity = size

        while self.ntimesteps > self.capacity:
            index = min(self.obs.keys())

            self.obs.pop(index)
            self.nxtobs.pop(index)
            self.actions.pop(index)
            self.rewards.pop(index)
            self.terminated.pop(index)
            self.truncated.pop(index)
            self.steps.pop(index)

    def statistics(self):
        episodes = [self.sample(ep).tonumpy() for ep in self.stopped]
        ep_returns = [sum(sample.rewards) for sample in episodes]
        mean, std = np.mean(ep_returns), np.std(ep_returns)

        return mean, std

    def dataset(self):
        obs = np.asarray([o for seq in self.obs.values() for o in seq], dtype=np.float32)
        actions = np.asarray([a for seq in self.actions.values() for a in seq], dtype=np.float32)
        nxtobs = np.asarray([nxt for seq in self.nxtobs.values() for nxt in seq], dtype=np.float32)
        rewards = np.asarray([r for seq in self.rewards.values() for r in seq], dtype=np.float32)
        terminated = np.asarray([t for seq in self.terminated.values() for t in seq], dtype=np.float32)
        truncated = np.asarray([t for seq in self.truncated.values() for t in seq], dtype=np.float32)
        steps = np.asarray([s for seq in self.steps.values() for s in seq], dtype=np.float32)

        return Sample(obs, actions, nxtobs, rewards, terminated, truncated, steps)

    @property
    def stopped(self):
        truncated = [idx for idx in self.episodes if bool(self.truncated[idx][-1])]
        return truncated

    @property
    def nstopped(self):
        return len(self.stopped)

    @property
    def episodes(self):
        return list(self.obs.keys())

    @property
    def nepisodes(self):
        return len(self.episodes)

    @property
    def ntimesteps(self):
        return sum([len(lst) for lst in self.obs.values()])


