import os
import math
import json
import time
import argparse
from typing import Type, Callable
from functools import partial

import gym
import wandb
import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

import pdecontrol.architectures as pdemodels
from pdegym.common.transforms import (
    BatchTransform,
    Normalize,
    SampleTransform,
    Operation,
)
from pdecontrol import visualize as visual
from pdecontrol.surrogates.common import schedulers
from pdecontrol.mbrl.types import Sample
from pdecontrol.callbacks import EvalLogCallback, VisCallback
from pdecontrol.surrogates.phyloss import phyloss
from pdecontrol.surrogates.utils import ignore_extra_keywords
from pdecontrol.surrogates.training import PDETrainingModule
from pdecontrol.surrogates.common.datamodule import PDEDataModule


parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str)
parser.add_argument("--env_id", type=str)
parser.add_argument("--factory", type=str)
parser.add_argument("--untransformed", action="store_true")
parser.add_argument("--data", type=str)
parser.add_argument("--target_length", type=int, default=30)
parser.add_argument("--splits", type=int, default=5)
parser.add_argument("--total", type=float, default=1.0)
parser.add_argument("--val", type=float, default=0.2)
parser.add_argument("--loss", type=str, default="MSELoss")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--offline", action="store_true")
parser.add_argument("--store", action="store_true")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--model", type=str, default="{}")
parser.add_argument("--surrogate", type=str, default="{}")
parser.add_argument("--training", type=str, default="{}")
parser.add_argument("--curriculum", type=str, default="{}")
parser.add_argument("--trainer", type=str, default="{}")
args = parser.parse_args()


if __name__ == "__main__":

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_SILENT"] = "true"

    env = gym.make(args.env_id, new_step_api=True, disable_env_checker=True)
    delta = env.scenario["cfg_steps"] * env.scenario["dt"]

    data: TensorDataset = torch.load(args.data)

    kfold = KFold(args.splits, shuffle=True, random_state=args.seed)
    index = math.ceil(args.total * len(data))
    index = torch.arange(index)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(index)):
        train_size = math.ceil((1.0 - args.val) * len(train_idx))
        train_idx, val_idx = train_idx[: train_size], train_idx[train_size :]  
        train_idx, val_idx, test_idx = torch.as_tensor(train_idx), torch.as_tensor(val_idx), torch.as_tensor(test_idx)

        data: TensorDataset = torch.load(args.data)
        data = Sample(*data.tensors).tonumpy()

        # Fit scaling transforms to training data.
        oscaling = Normalize(aggregate=True, batched=True)
        ascaling = Normalize(aggregate=True, batched=True)
        forcing = BatchTransform(env.forcing)
        pdescaling = Normalize(aggregate=True, batched=True)
        undscaling = Normalize(aggregate=True, batched=True)

        # Sample transform applied to scale actions & observations.
        atransf = ascaling if args.untransformed else Operation([forcing, pdescaling])
        stransf = SampleTransform(oscaling, atransf)

        obs, actions, nxtobs, *_ = data
        obs, actions, nxtobs = obs[train_idx], actions[train_idx], nxtobs[train_idx]
        bsize, nsteps, nchannels, height = obs.shape
        obs = obs.reshape(bsize * nsteps, nchannels, height)
        bsize, nsteps, nchannels, height = actions.shape
        actions = actions.reshape(bsize * nsteps, nchannels, height)
        bsize, nsteps, nchannels, height = nxtobs.shape
        nxtobs = nxtobs.reshape(bsize * nsteps, nchannels, height)
        
        # Update observation and action scaling.
        oscaling.update(obs)
        ascaling.update(actions)
        pdescaling.update(forcing(actions))

        # Scale changes between state transition.
        deltas = oscaling(nxtobs) - oscaling(obs)
        undscaling.update(deltas / delta)

        # Load the factory class using importlib.
        factory = getattr(pdemodels, args.factory)()

        # Use JSON to parse the CLI arguments passed as dictionaries.
        model = json.loads(args.model)
        surrogate = json.loads(args.surrogate)
        training = json.loads(args.training)
        curriculum = json.loads(args.curriculum)
        trainer = json.loads(args.trainer)

        # Set target sequence length for training.
        training["target_length"] = args.target_length

        # Save configuration and log results with wandb.
        config = {
            "factory": args.factory,
            "model": {**factory.defaults.model, **model},
            "surrogate": {**factory.defaults.surrogate, **surrogate},
            "training": {**factory.defaults.training, **training},
            "curriculum": {**factory.defaults.curriculum, **curriculum},
            "trainer": {**factory.defaults.trainer, **trainer},
            "loss": args.loss,
        }

        target = training.get("target_length")
        run = wandb.init(project=args.project, config=config)
        wandb.run.name = f"{wandb.run.id}-{args.factory}-Data-{args.total}-Target-{target}-CV-{fold}"
        wandb.run.save()

        logger = WandbLogger()

        # Load the physical loss class or e,g. the MSELoss.
        env = gym.make(args.env_id, new_step_api=True)
        loss: Type[Callable] = getattr(phyloss, args.loss)
        loss = ignore_extra_keywords(loss)(**env.unwrapped.scenario, reduction="none")

        # Use the factory class to build surrogate & training components.
        model = factory.model(**config["model"])
        surrogate = factory.surrogate(**model, **config["surrogate"], delta=delta, dscaling=undscaling.Inverse)
        tstep = env.cfg_steps * env.dt

        module = PDETrainingModule(
            surrogate=surrogate,
            loss=loss,
            env=env,
            stransf=stransf,
            tstep=tstep,
            delta=delta,
            undscaling=undscaling,
            **config["training"],
        )

        logger.watch(module, log="all", log_graph=True, log_freq=50)
        
        delta_plot = partial(visual.spatial, ["outdeltas", "deltas"])
        plotting={"PDE Plot": visual.pdeplot, "Delta Plot": delta_plot}
        viscallback = VisCallback(plotting=plotting, stransf=stransf, reward_func=env.reward_func)

        early_stopping = EarlyStopping(
            monitor="Val. Loss",
            patience=config["training"]["patience"],
        )

        evalcallback = EvalLogCallback()
        lr_monitor = LearningRateMonitor()

        trainer = pl.Trainer(
            logger=logger,
            enable_model_summary=True,
            enable_progress_bar=True,
            reload_dataloaders_every_n_epochs=1,
            callbacks=[early_stopping, viscallback, evalcallback, lr_monitor],
            log_every_n_steps=1,
            **config["trainer"],
        )

        # Load curriculum scheduler with importlib.
        curriculum = schedulers.ConstantLengthScheduler(length=args.target_length)

        datamodule = PDEDataModule(
            data=data,
            train=train_idx,
            val=val_idx,
            test=test_idx,
            stransf=stransf,
            curriculum=curriculum,
            **config["training"],
        )

        start = time.time()
        trainer.fit(module, datamodule=datamodule)
        stop = time.time()
        wandb.log({"Training Time": stop - start})

        trainer.test(module, datamodule=datamodule)

        if args.store:
            checkpoint = {"oscaling": oscaling, "ascaling": ascaling, "forcing": forcing, "pdescaling": pdescaling, "undscaling": undscaling, "state_dict": module.surrogate.state_dict()}
            torch.save(module.surrogate.state_dict(), f"{wandb.run.name}/model.pl")
            artifact = wandb.Artifact(f"{wandb.run.name}", type="model")
            run.log_artifact(artifact)
            
        run.finish()
