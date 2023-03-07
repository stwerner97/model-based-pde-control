from collections import defaultdict
from typing import Dict, Callable, Iterable
from pathlib import Path

import wandb
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from pdegym.common.transforms import SampleTransform


class VisCallback(pl.callbacks.Callback):
    def __init__(
        self, plotting: Dict = {}, stransf: SampleTransform = None, reward_func: Callable = None, log_freq: int = 1
    ):
        super().__init__()

        self.plotting = plotting
        self.stransf = stransf if stransf is not None else SampleTransform()
        self.reward_func = reward_func
        self.log_freq = log_freq

        self.num_val_batch_end = 0
        self.num_train_batch_end = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, bidx, dataloader_idx
    ):
        self.num_val_batch_end += 1
        if self.num_val_batch_end % self.log_freq != 0:
            return

        num_batches = trainer.num_val_batches[dataloader_idx]
        
        # Log on last batch of validation epoch.
        if bidx < num_batches - 1:
            return

        self.plot_batch(trainer, batch, outputs, stage="Val.")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, bidx
    ):
        self.num_train_batch_end += 1
        if self.num_train_batch_end % self.log_freq != 0:
            return

        num_batches = trainer.num_training_batches
        
        # Log on last batch of training epoch.
        if bidx < num_batches - 1:
            return

        self.plot_batch(trainer, batch, outputs, stage="Train")

    def plot_batch(self, trainer, batch, outputs, stage):
        outputs = {key: val.detach().numpy() for key, val in outputs.items()}

        # Sample random sequence of the batch to plot.
        obs, opred, actions = outputs["states"], outputs["outputs"], outputs["actions"]
        index = np.random.randint(actions.shape[0])
        actions, obs, opred = actions[index], obs[index], opred[index]
        deltas, outdeltas = np.diff(obs, axis=0), np.diff(opred, axis=0)

        # Undo applied transforms before plotting.
        obs = self.stransf.otransf.Inverse(obs)
        opred = self.stransf.otransf.Inverse(opred)
        actions = self.stransf.atransf.Inverse(actions)

        deltas, outdeltas = np.diff(obs, axis=0), np.diff(opred, axis=0)

        rewards = np.asarray([self.reward_func(o, a) for o, a in zip(obs, actions)])
        rpred = np.asarray([self.reward_func(o, a) for o, a in zip(opred, actions)])

        data = {"obs": obs, "actions": actions, "opred": opred, "rewards": rewards, "rpred": rpred, "deltas": deltas, "outdeltas": outdeltas}
        data = {key: np.squeeze(val, axis=1) if val.ndim > 2 else val for key, val in data.items()}

        for name, plotfnc in self.plotting.items():
            image = plotfnc(**data)
            trainer.logger.log_image(f"{stage} {name}", [image])


class EvalLogCallback(pl.callbacks.Callback):
    def __init__(self, Nstore=20):
        super().__init__()

        self.outputs = []
        self.batch_sizes = []

        self.Nstore = Nstore

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, bidx, dataloader_idx
    ):
        obs, *_ = batch

        # Cache batch outputs and batch sizes. 
        self.outputs.append(outputs)
        self.batch_sizes.append(obs.size(0))

    def on_test_epoch_end(self, trainer, pl_module):

        # Save ground-truth states, predictions and actions.
        states = np.concatenate([output.pop("states") for output in self.outputs])
        outputs = np.concatenate([output.pop("outputs") for output in self.outputs])
        actions = np.concatenate([output.pop("actions") for output in self.outputs])

        # Select a random subset of to-be-stored sequences.
        states, actions, outputs = states[:self.Nstore], actions[:self.Nstore], outputs[:self.Nstore]

        Path(f"{wandb.run.id}/test").mkdir(parents=True, exist_ok=True)

        np.savez(f"{wandb.run.id}/test/test.npz", states=states, outputs=outputs, actions=actions)
        artifact = wandb.Artifact(name=f"{wandb.run.id}-test", type='dataset')
        artifact.add_file(f"{wandb.run.id}/test/test.npz")
        wandb.run.log_artifact(artifact)

        # Aggregate loss values of all batches.
        aggregated = defaultdict(float)
        for output, bsize in zip(self.outputs, self.batch_sizes):
            for key, value in output.items():
                aggregated[key] += bsize * value

        for key, value in aggregated.items():
            aggregated[key] = value / sum(self.batch_sizes)
            
        table = {key: list(value) for key, value in aggregated.items() if isinstance(value, Iterable)}
        table = pd.DataFrame.from_dict(table)
        table = wandb.Table(dataframe=table)
        wandb.log({"Time Table": table})

        scalars = {key: value for key, value in aggregated.items() if not isinstance(value, Iterable)}
        wandb.log(scalars)

        # Clear caches.
        self.outputs = []
        self.batch_sizes = []