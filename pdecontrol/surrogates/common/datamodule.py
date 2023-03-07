import os
from typing import List

import pytorch_lightning as pl
from pdecontrol.surrogates.common.schedulers import FuncScheduler, LinearScheduler, Scheduler

from pdegym.common.transforms import SampleTransform
from pdecontrol.surrogates.common.dataset import PDEDataLoader, SubSeqDataset


class PDEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data,
        train: List[int],
        val: List[int] = None,
        test: List[int] = None,
        bootstrapping:bool = True,
        stransf: SampleTransform = None,
        curriculum: Scheduler = None,
        iteration: int = 0,
        tau: int = 5,
        stride: int = None,
        target_length: int = None,
        shuffle: bool = True,
        batch_size: int = 128,
        **kwargs
    ):
        super().__init__()

        self.data = data
        self.train = train
        self.val = val
        self.test = test
        self.bootstrapping = bootstrapping
        self.stransf = stransf
        self.curriculum = curriculum
        self.tau = tau
        self.stride = stride
        self.target_length = target_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.iteration = iteration

        if self.curriculum is None:
            self.curriculum = FuncScheduler(steptype="epoch", func=lambda *_: 1)

    def train_dataloader(self):

        # Schedule the number of extrapolation steps with curriculum learning.
        K = int(
            self.curriculum(
                self.iteration, self.trainer.current_epoch, self.trainer.global_step
            )
        )

        data = SubSeqDataset(
            data=self.data,
            subsamples=self.train,
            length=self.tau + K,
            stride=self.stride,
            bootstrapping=self.bootstrapping,
            stransf=self.stransf,
        )

        return PDEDataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=PDEDataLoader.sample_collate,
        )

    def val_dataloader(self):

        # Schedule the number of extrapolation steps with curriculum learning.
        K = int(
            self.curriculum(
                self.iteration, self.trainer.current_epoch, self.trainer.global_step
            )
        )

        data = SubSeqDataset(
            data=self.data,
            subsamples=self.val,
            length=self.tau + K,
            stride=self.stride,
            bootstrapping=self.bootstrapping,
            stransf=self.stransf,
        )

        return PDEDataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=PDEDataLoader.sample_collate,
        )

    def test_dataloader(self):

        data = SubSeqDataset(
            data=self.data,
            subsamples=self.test,
            length=self.tau + self.target_length,
            stride=self.tau,
            bootstrapping=False,
            stransf=self.stransf,
        )

        return PDEDataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=PDEDataLoader.sample_collate,
        )
