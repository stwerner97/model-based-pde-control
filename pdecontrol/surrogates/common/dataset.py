from collections import defaultdict
import math
import bisect
from itertools import islice
from typing import List, Tuple

import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from pdegym.common.transforms import SampleTransform
from pdecontrol.mbrl.types import Sample


class SubSeqDataset(Dataset):
    def __init__(
        self,
        data: Tuple,
        subsamples: List[int] = None,
        length: int = 1,
        stride: int = None,
        bootstrapping: bool = True,
        bounds: Tuple[int, int] = (0, 0),
        stransf: SampleTransform = None,
    ):
        super().__init__()

        (
            self.obs,
            self.actions,
            self.nxtobs,
            self.rewards,
            self.terminated,
            self.truncated,
            self.steps,
        ) = data
        
        self.subsamples = subsamples
        self.length = length
        self.stride = stride
        self.bootstrapping = bootstrapping
        self.lower, self.upper = bounds
        self.stransf = stransf


        if self.subsamples is None and isinstance(self.obs, defaultdict):
            self.subsamples = list(self.obs.keys())

        elif self.subsamples is None:
            self.subsamples = list(np.arange(self.obs.shape[0]))
        
        # Set stride to length of sequences (i.e. non-overlapping sampling).
        if self.stride is None:
            self.stride = length

        # Build search index for sub-sequence samples.
        self.index = np.cumsum(self.count_sub_seqs(self.length, self.stride))

        # Prepare bootstrapping index and mapping.
        self.boots_index = np.cumsum(self.count_sub_seqs(self.length, 1))
        self.boots_mapping = np.random.randint(low=0, high=np.max(self.boots_index, initial=0).astype(np.int32), size=len(self))
        
    def __getitem__(self, idx):

        assert(idx < len(self))

        idx = idx if not self.bootstrapping else self.boots_mapping[idx]
        index =  self.index if not self.bootstrapping else self.boots_index
        stride = self.stride if not self.bootstrapping else 1

        key = bisect.bisect_right(index, idx)

        # Get index of sub-sequence within sequence.
        offset = index[key - 1] if key - 1 >= 0 else 0
        sidx = (idx - offset) * stride + self.lower
        
        # Get index of sequence key.
        bidx = self.subsamples[key]

        obs = list(islice(self.obs[bidx], sidx, sidx + self.length))
        obs = np.asarray(obs, dtype=np.float32)
        actions = list(islice(self.actions[bidx], sidx, sidx + self.length))
        actions = np.asarray(actions, dtype=np.float32)
        nxtobs = list(islice(self.nxtobs[bidx], sidx, sidx + self.length))
        nxtobs = np.asarray(nxtobs, dtype=np.float32)
        rewards = list(islice(self.rewards[bidx], sidx, sidx + self.length))
        rewards = np.asarray(rewards, dtype=np.float32)
        terminated = list(islice(self.terminated[bidx], sidx, sidx + self.length))
        terminated = np.asarray(terminated, dtype=np.bool8)
        truncated = list(islice(self.truncated[bidx], sidx, sidx + self.length))
        truncated = np.asarray(truncated, dtype=np.bool8)
        steps = list(islice(self.steps[bidx], sidx, sidx + self.length))
        steps = np.asarray(steps, dtype=np.int32)

        sample = Sample(obs, actions, nxtobs, rewards, terminated, truncated, steps)

        if self.stransf:
            sample = self.stransf(sample)

        return sample.totorch()

    def __len__(self):
        return np.max(self.index, initial=0).astype(np.int32)

    def count_sub_seqs(self, length, stride):
        index = [len(self.obs[idx]) - self.lower - self.upper for idx in self.subsamples]
        index = [self.count_seq_sub_seqs(nelems, length, stride) for nelems in index]
        return index

    def count_seq_sub_seqs(self, nelems, length, stride):
        return max(math.floor((nelems - length) / stride) + 1, 0)

    @property
    def max_seq_length(self):
        return max([len(self.obs[idx]) - self.lower - self.upper for idx in self.subsamples])


class StartingStateDataset(ConcatDataset):
    def __init__(
        self,
        data: Tuple,
        subsamples: List[int] = None,
        length: int = 1,
        stride: int = None,
        bootstrapping: bool = False,
        bounds: Tuple[int, int] = (0, 0),
        stransf: SampleTransform = None,
    ):

        # After the initial ``tau`` episode steps: sample sequences of length ``tau``.
        sequences = SubSeqDataset(
            data=data,
            subsamples=subsamples,
            length=length,
            stride=stride,
            bootstrapping=bootstrapping,
            bounds=bounds,
            stransf=stransf
        )
        starting = [sequences]

        # At the beginning of the episode: sample sequences with smaller warmup phase than ``tau``.
        lower, upper = bounds
        lengths = list(1 + np.arange(length))
        strides = [length - lngth + 1 for lngth in lengths]

        for length, stride in zip(lengths, strides):
            dataset = SubSeqDataset(
                data=data,
                subsamples=subsamples,
                length=length,
                stride=stride,
                bootstrapping=bootstrapping,
                bounds=(lower, upper + sequences.max_seq_length - length),
                stransf=stransf
            )
            starting.append(dataset)

        super().__init__(starting)


class PDEDataLoader(DataLoader):
    @staticmethod
    def sample_collate(samples):
        samples = [tuple(sample) for sample in samples]
        return default_collate(samples)

    @staticmethod
    def padding_collate(samples):
        obs = [sample.obs for sample in samples]
        actions = [sample.actions for sample in samples]
        nxtobs = [sample.nxtobs for sample in samples]
        rewards = [sample.rewards for sample in samples]
        terminated = [sample.terminated for sample in samples]
        truncated = [sample.truncated for sample in samples]
        steps = [sample.steps for sample in samples]

        obs = PDEDataLoader.repeat_padding(obs, dim=0) 
        actions = PDEDataLoader.repeat_padding(actions, dim=0) 
        nxtobs = PDEDataLoader.repeat_padding(nxtobs, dim=0)
        rewards = PDEDataLoader.repeat_padding(rewards, dim=0)
        terminated = PDEDataLoader.repeat_padding(terminated, dim=0)
        truncated = PDEDataLoader.repeat_padding(truncated, dim=0)
        steps = PDEDataLoader.repeat_padding(steps, dim=0)

        sample = Sample(obs, actions, nxtobs, rewards, terminated, truncated, steps)
        return sample

    @staticmethod
    def repeat_padding(tensors, dim=0):
        sizes = [tensor.size(dim) for tensor in tensors]
        max_size = max(sizes)

        outputs = []

        for size, tensor in zip(sizes, tensors):
            nrepeats = max_size - size

            repeated = torch.index_select(tensor, dim=dim, index=torch.as_tensor(0))
            repeated = torch.repeat_interleave(repeated, nrepeats, dim=dim)
            tensor = torch.cat((repeated, tensor), dim=dim)
            outputs.append(tensor)

        return torch.stack(outputs)


class ResampleDataLoader(DataLoader):
    """Taken from https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)

        except StopIteration:
            self.iterator = super().__iter__()
            batch = next(self.iterator)

        return batch
