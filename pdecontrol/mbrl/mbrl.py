import time
from pathlib import Path
from functools import partial
from argparse import Namespace
from typing import List, Type, Callable

import gym
import json
import wandb
import numpy as np
import pytorch_lightning as pl
from tabulate import tabulate
from gym.vector import VectorEnv
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping
from pdecontrol.surrogates.surrogate import PDEEnsemble


from pdegym.common.transforms import BatchTransform, Normalize, SampleTransform, ScaleTransform, SensorTransform
from pdegym.common.vec_wrappers import (
    TransformObsWrapper,
    StoreNObsVecWrapper,
    StoreNActionsVecWrapper,
    TransformActionWrapper,
)
import pdecontrol.visualize as visual
from pdecontrol.callbacks import VisCallback
from pdecontrol.sac.sac import SAC
from pdecontrol.mbrl import utils
from pdecontrol.mbrl.types import PDETrainer
from pdecontrol.mbrl.replay import ExperienceReplay
from pdecontrol.mbrl.callbacks import VisPDECallback
from pdecontrol.mbrl.worker import Worker, PDEEnvStack
from pdecontrol.mbrl.world.world import WorldVecEnv
from pdecontrol.mbrl.world.wrappers import BaseWorldVecEnvWrapper
from pdecontrol.surrogates.phyloss import phyloss
from pdecontrol.surrogates.training import PDETrainingModule
from pdecontrol.surrogates.common.schedulers import Scheduler
from pdecontrol.surrogates.factory import PDESurrogateFactory
from pdecontrol.surrogates.common.datamodule import PDEDataModule
from pdecontrol.surrogates.common.dataset import PDEDataLoader, SubSeqDataset, StartingStateDataset
from pdecontrol.surrogates.utils import ignore_extra_keywords


class PDEModelBasedController:
    HEADERS = [
        "Iterations",
        "Time",
        "Num. Sur. Upd.",
        "Num. Pol. Upd.",
        "Num. Steps Sampled",
        "Avg. Eval. Ep. Return",
        "Avg. World Ep. Return",
        "Horizon",
        "World Buffer Samples",
        "Train Loss",
        "Val. Loss",
        "Sur./K",
        "SAC/Qloss",
        "SAC/PolicyLoss",
    ]

    def __init__(
        self,
        env_id: str,
        factory: PDESurrogateFactory,
        config: Namespace,
        args: Namespace,
    ):
        self.factory = factory
        self.config = config
        self.args = args

        self.pllogger = WandbLogger()

        self.env = gym.make(env_id, new_step_api=True)

        # Create vector environments used for experience collection.
        self.envs: VectorEnv = gym.vector.make(
            env_id, num_envs=self.args.cpus, new_step_api=True
        )
        self.eval_envs: VectorEnv = gym.vector.make(
            env_id, num_envs=self.args.cpus, new_step_api=True
        )

        # Set up parameters values and runtime variables. 
        self.samples_per_iteration = self.args.cpus * self.args.rollout_length
        self.num_pol_updates_per_iteration = int(self.args.policy_train_steps_per_sample * self.samples_per_iteration)
        self.sur_train_freq = int(self.args.surrogate_train_freq / self.samples_per_iteration)
        self.iteration, self.num_ensemble_updates, self.num_pol_updates = 0, 0, 0
        self.tau = self.config.training["tau"]

        # Build scheduler determining the length of model rollouts.
        self.args.rollout_length_schedule = json.loads(self.args.rollout_length_schedule)
        self.schedule = Scheduler.factory(config=self.args.rollout_length_schedule)

        # Build scheduler determining the length of training subsequences.
        self.curriculum = Scheduler.factory(config=self.config.curriculum)

        # Set up data & connector transformations.
        self.setup_transforms()

        # Define callbacks to track & visualize results.
        self.setup_callbacks()

        # Set up the dynamics models.
        self.modules: List[PDETrainingModule] = [self.setup_surrogate() for _ in range(self.args.num_dynamics_models)]

        # Set up the ensemble of dynamics models.
        self.ensemble = PDEEnsemble(modules=self.modules, num_elites=self.args.num_elite_models)

        # Set up a trainer and callbacks for each model of the ensemble.
        self.trainers: List[PDETrainer] = [self.setup_trainer() for _ in range(self.args.num_dynamics_models)]

        # Set up the wrapped & surrogate environments.
        self.setup_wrapped_envs()
        self.setup_world_envs()

        # Set up replay buffers for real and imaginary experience.
        self.replay = ExperienceReplay(capacity=self.args.capacity)
        self.world_replay = ExperienceReplay(capacity=self.imaginary_buffer_capacity)

        # Set up rollout workers for envs. & evaluation envs.
        self.worker = Worker(self.stack)
        self.eval_worker = Worker(self.eval_stack, callbacks=[self.eval_vis])

        # Set up rollout workers for world env. and eval. world env.
        self.world_worker = Worker(self.world_stack, callbacks=[self.world_vis])
        self.eval_world_worker = Worker(self.eval_world_stack)

        # Agent interacts with PDE / world environment via the surrogate.
        self.agent = SAC(
            self.stack.envs.single_observation_space,
            self.stack.envs.single_action_space,
            config=args,
        )

        # Define rollout stopping conditions.
        self.setup_stopping_conditions()

        # Set up output folder for evaluation data.
        Path(f"{wandb.run.id}/evaluation").mkdir(parents=True)

    def setup_transforms(self) -> None:
        # Set up scaling of observations.
        self.oscaling = ScaleTransform(batched=True, aggregate=True, frozen=False)

        # Set up scaling of actions according to their bounds. 
        low = self.envs.single_action_space.low
        high = self.envs.single_action_space.high
        low, high = low[np.newaxis, ...], high[np.newaxis, ...]
        self.ascaling = ScaleTransform(bounds=(low, high), aggregate=True, frozen=True, batched=True).Inverse

        # Set up transformation to external forcing term.
        self.forcing = BatchTransform(self.env.forcing)

        # Set up scaling of the external forced terms.
        low = self.envs.single_action_space.low[np.newaxis, ...]
        high = self.envs.single_action_space.high[np.newaxis, ...]
        low = np.squeeze(self.forcing(low), axis=0)
        high = np.squeeze(self.forcing(high), axis=0)
        self.pdescaling = ScaleTransform(bounds=(low, high), scale=(-1, 1), aggregate=True, frozen=True)
        self.pdescaling = BatchTransform(self.pdescaling)

        # Set up normalization of scaled state changes.
        self.undscaling = Normalize(aggregate=True, batched=True)

        # Set up sensor transformation for the agent's observations.
        self.agent_sensor = BatchTransform(SensorTransform(stride=1))
        
        # Set up transform to downsample state variables of the environment.
        self.world_sensor = BatchTransform(SensorTransform(stride=1))

        # Set up connector sample transform between replay and agent.
        self.replay_to_agent = SampleTransform(
            otransf=[self.oscaling, self.agent_sensor],
            atransf=self.ascaling.Inverse,
        )

        # Set up connector to transform PDE samples to world model samples.
        otransf = [self.oscaling, self.world_sensor]
        atransf = [self.forcing, self.pdescaling, self.world_sensor]
        self.replay_to_world = SampleTransform(otransf, atransf)

        # Set up connector to transform world samples.
        self.world_replay_to_agent = SampleTransform(atransf=self.ascaling.Inverse)

    def setup_callbacks(self) -> None:
        # Set up callback to visualize training samples.
        delta_plot = partial(visual.spatial, ["outdeltas", "deltas"])
        plotting = {"PDE Plot": visual.pdeplot, "Delta Plot": delta_plot}
        self.viscallback = VisCallback(
            plotting=plotting,
            reward_func=self.env.reward_func,
            log_freq=self.args.logging_freq,
        )

        # Set up callback to visualize world env. rollouts.
        log_freq = np.ceil(self.num_world_rollouts / self.args.model_rollouts_batch_size)
        self.world_vis = VisPDECallback(
            plotting={"World Env. Episode": visual.epplot},
            log_freq=int(self.args.logging_freq * log_freq)
        )

        # Set up callback to visualize eval. rollouts.
        self.eval_vis = VisPDECallback(plotting={"Eval. Episode": visual.epplot})


    def setup_surrogate(self) -> PDETrainingModule:
        # Set up the data-driven training components (data loss & module).
        DataLoss: Type[Callable] = getattr(phyloss, self.config.loss)
        dataloss = ignore_extra_keywords(DataLoss)(
            **self.env.unwrapped.scenario, reduction="none"
        )

        # Build surrogate model and training components from passed factory.
        delta = self.env.scenario["cfg_steps"] * self.env.scenario["dt"]
        model = self.factory.model(**self.env.scenario, **self.config.model)
        surrogate = self.factory.surrogate(
            delta=delta,
            dscaling=self.undscaling.Inverse,
            tau=self.tau,
            **self.env.scenario,
            **self.config.surrogate,
            **model,
        )

        training_config = "initial" if self.iteration <= 0 else "iterations"
        training_config = self.config.training[training_config]

        # Create training module for PDE surrogate.
        module = PDETrainingModule(
            surrogate=surrogate,
            loss=dataloss,
            tau=self.tau,
            stransf=self.replay_to_world.Inverse,
            tstep=self.env.cfg_steps * self.env.dt,
            delta=self.env.cfg_steps * self.env.dt,
            undscaling=self.undscaling,
            **training_config,
        )

        return module

    def setup_stopping_conditions(self) -> None:
        # Set up stopping conditions used to terminate rollouts.
        self.warmup = lambda ts, _: ts >= self.args.learning_starts
        self.sampling = lambda ts, _: ts >= self.samples_per_iteration
        self.eval_stop = lambda _, ep: ep >= self.args.num_eval_episodes

        # Set up stopping condition used to terminate model-based rollouts.
        self.world_stop = lambda _, eps: eps >= self.num_world_rollouts
        self.world_eval_stop = lambda ts, eps: eps >= 1

    def setup_wrapped_envs(self) -> None:
        # Build wrapped environment stack for experience collection.
        ostore = StoreNObsVecWrapper(self.envs, num_steps=1)
        self.envs = TransformObsWrapper(ostore, self.oscaling, frozen=False)
        self.envs = TransformObsWrapper(self.envs, self.world_sensor)
        world_wrapper = BaseWorldVecEnvWrapper(
            env=self.envs,
            surrogate=self.ensemble,
            tstep=self.env.cfg_steps * self.env.dt,
        )
        self.envs = TransformObsWrapper(world_wrapper, self.agent_sensor)
        astore = StoreNActionsVecWrapper(self.envs, num_steps=1)
        self.envs = TransformActionWrapper(astore, self.ascaling, frozen=True)
        self.stack = PDEEnvStack(
            envs=self.envs, ostore=ostore, astore=astore, world_wrapper=world_wrapper
        )

        # Build wrapped environment stack for the evaluation.
        ostore = StoreNObsVecWrapper(self.eval_envs, num_steps=1)
        self.eval_envs = TransformObsWrapper(ostore, self.oscaling, frozen=True)
        self.eval_envs = TransformObsWrapper(self.eval_envs, self.world_sensor)
        world_wrapper = BaseWorldVecEnvWrapper(
            env=self.eval_envs,
            surrogate=self.ensemble,
            tstep=self.env.cfg_steps * self.env.dt,
        )
        self.eval_envs = TransformObsWrapper(world_wrapper, self.agent_sensor)
        astore = StoreNActionsVecWrapper(self.eval_envs, num_steps=1)
        self.eval_envs = TransformActionWrapper(astore, self.ascaling, frozen=True)
        self.eval_stack = PDEEnvStack(
            envs=self.eval_envs,
            ostore=ostore,
            astore=astore,
            world_wrapper=world_wrapper,
        )

    def setup_world_envs(self) -> None:
        horizon = int(self.schedule(iteration=self.iteration))

        self.world = WorldVecEnv(
            surrogate=self.ensemble,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            max_episode_steps=self.env.unwrapped.max_episode_steps,
            stransf=self.replay_to_world.Inverse,
            reward_func=self.env.reward_func,
            num_envs=self.args.model_rollouts_batch_size,
            horizon=horizon,
            tstep=self.env.cfg_steps * self.env.dt,
        )

        self.eval_world = WorldVecEnv(
            surrogate=self.ensemble,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            max_episode_steps=self.env.unwrapped.max_episode_steps,
            stransf=self.replay_to_world.Inverse,
            reward_func=self.env.reward_func,
            num_envs=1,
            horizon=horizon,
            tstep=self.env.cfg_steps * self.env.dt,
        )

        # Build wrapper stack for surrogate data collection.
        ostore = StoreNObsVecWrapper(self.world, num_steps=1)
        self.world_env = TransformObsWrapper(ostore, self.agent_sensor)
        self.world_env = TransformActionWrapper(self.world_env, self.world_sensor)
        self.world_env = TransformActionWrapper(self.world_env, self.pdescaling, frozen=True)
        self.world_env = TransformActionWrapper(self.world_env, self.forcing, frozen=True)
        self.world_env = TransformActionWrapper(self.world_env, self.ascaling, frozen=True)
        astore = StoreNActionsVecWrapper(self.world_env, num_steps=1)
        self.world_stack = PDEEnvStack(
            envs=astore, ostore=ostore, astore=astore, world_wrapper=None
        )

        # Build wrapper stack for surrogate evaluation.
        ostore = StoreNObsVecWrapper(self.eval_world, num_steps=1)
        self.eval_world_env = TransformObsWrapper(ostore, self.agent_sensor)
        self.eval_world_env = TransformActionWrapper(self.eval_world_env, self.world_sensor)
        self.eval_world_env = TransformActionWrapper(self.eval_world_env, self.pdescaling, frozen=True)
        self.eval_world_env = TransformActionWrapper(self.eval_world_env, self.forcing, frozen=True)
        self.eval_world_env = TransformActionWrapper(self.eval_world_env, self.ascaling, frozen=True)
        astore = StoreNActionsVecWrapper(self.eval_world_env, num_steps=1)
        self.eval_world_stack = PDEEnvStack(
            envs=astore, ostore=ostore, astore=astore, world_wrapper=None
        )

    def setup_trainer(self) -> PDETrainer:
        # Determine trainer configuration at the initial iteration.
        training_config = self.config.training["initial"]
        trainer_config= self.config.trainer["initial"]
        trainer_config= {**trainer_config,  **{"max_steps": 0, "min_steps": 0}}
        
        # Set up early stopping on validation loss.
        early_stopping = EarlyStopping(
            monitor="Val. Loss",
            patience=training_config["patience"],
        )

        # The trainer reloads dataloaders after each episode to subsample sequences.
        trainer = pl.Trainer(
            logger=self.pllogger,
            enable_model_summary=False,
            enable_progress_bar=True,
            reload_dataloaders_every_n_epochs=1,
            callbacks=[early_stopping, self.viscallback],
            log_every_n_steps=1,
            **trainer_config
        )

        return PDETrainer(trainer=trainer, early_stopping=early_stopping)

    def reset_trainer(self, trainer: PDETrainer) -> None:
        phase = "initial" if self.iteration <= 0 else "iterations"
        training_config = self.config.training[phase]
        trainer_config = self.config.trainer[phase]

        # Reset the number of violated violation checks.
        trainer.early_stopping.wait_count = 0
        trainer.early_stopping.patience = training_config["patience"]

        # Set number of minimum / maximum gradient steps to take.
        trainer.trainer.fit_loop.max_steps = trainer.trainer.global_step + trainer_config.get("max_steps", 0)
        trainer.trainer.fit_loop.min_steps = trainer.trainer.global_step + trainer_config.get("min_steps", 0)

        trainer.trainer.should_stop = False

    def learn(self):
        wandb.log({"Start": time.time()}, commit=False)

        # Initial exploration with random action sampling policy.
        self.stack.world_wrapper.disable()
        explore = utils.RandomAgent(self.envs.action_space)
        rollout = self.worker.rollout(explore, self.warmup)
        self.replay.extend(rollout)
        self.stack.world_wrapper.enable()

        # Evaluate untrained policy & surrogate model.
        self.evaluate_policy(self.agent)

        while (
            self.num_steps_sampled
            < self.args.total_timesteps - self.args.learning_starts
        ):
            # Interact with env. and collect experience with policy.
            rollout = self.worker.rollout(self.agent, self.sampling)
            self.replay.extend(rollout)

            # Update surrogate models every `sur_train_freq` iterations.
            if self.iteration % self.sur_train_freq == 0:
                self.update_delta_transform()
                scores = [self.update_surrogate(module, trainer) for module, trainer in zip(self.ensemble.modules, self.trainers)]
                self.ensemble.update_elites(scores)

                self.num_ensemble_updates += 1
                wandb.log({"Num. Ensemble Updates": self.num_ensemble_updates}, commit=False)

            # Sample starting states for world env. from replay buffer.
            starting = StartingStateDataset(
                data=self.replay.data,
                length=self.tau,
                stride=1,
                bootstrapping=False,
                stransf=self.replay_to_world,
            )
            self.world.setup(starting)

            # Adjust traj. horizon of world env. rollouts.
            self.world.horizon = int(self.schedule(iteration=self.iteration))

            # Resize buffer dependent on the rollout horizon.
            self.world_replay.resize(self.imaginary_buffer_capacity)

            # Rollout policy in world env. and store samples in separate buffer.
            rollout = self.world_worker.rollout(self.agent, self.world_stop)
            self.world_replay.extend(rollout)

            # Reset worker after rollout (otherwise leads to inconsistencies after model updates).
            self.world_worker.reset()

            # Update the policy using artifically generated experience.
            self.update_policy()

            # Evaluate the agent's performance & world model's quality.
            if self.iteration % self.args.agent_eval_freq == 0:
                self.evaluate_policy(self.agent)
                self.evaluate_surrogate()
                self.log_world_stats()

            self.end_iteration()

            if self.iteration % self.args.status_report_freq == 0:
                self.summarize()

    def log_world_stats(self):
        mean, std = self.world_replay.statistics()
        wandb.log(
            {
                "Avg. World Rll. Return": mean,
                "Std. World. Rll. Return": std,
                "Avg. World Step Rew.": mean / self.world.horizon,
            },
            commit=False,
        )

    def evaluate_policy(self, policy):
        rollout = self.eval_worker.rollout(policy, self.eval_stop, deterministic=True)
        mean, std = rollout.statistics()
        wandb.log({"Avg. Eval. Ep. Return": mean, "Std. Eval. Ep. Return": std}, commit=True)

        # Save evaluation episodes as artifacts.
        obs, actions, _, rewards, *_ = rollout.dataset()
        np.savez(f"{wandb.run.id}/evaluation/eval_{self.iteration}.npz", obs=obs, actions=actions, rewards=rewards)
        artifact = wandb.Artifact(name=f"{wandb.run.id}-evaluation-eval-{self.iteration}", type='dataset')
        artifact.add_file(f"{wandb.run.id}/evaluation/eval_{self.iteration}.npz")
        wandb.run.log_artifact(artifact)

    def evaluate_surrogate(self, horizon=30):
        if not self.replay.stopped:
            return

        # Sample episode from replay buffer.
        index = np.random.choice(self.replay.stopped)
        sample = self.replay.sample(index)
        sample = sample.apply(lambda x: x.unsqueeze(0)).tonumpy()

        # Randomly choose a starting point for the evaluation.
        length = sample.obs.shape[1]
        start = np.random.randint(0, length - self.tau - horizon)

        # Warm-start episode in eval. world env. with states taken from the replay.
        starting = sample.apply(lambda x: x[:, start : start + self.tau])
        starting = SubSeqDataset(
            data=starting,
            length=self.tau,
            bootstrapping=False,
            stransf=self.replay_to_world
        )
        self.eval_world.setup(starting)
        self.eval_world.horizon = horizon

        # Repeat actions of sample episode in eval. world env.
        actions = self.replay_to_agent.atransf(np.squeeze(sample.actions, axis=0))
        actions = actions[
            np.newaxis, start + self.tau : start + self.tau + horizon
        ]
        eval_agent = utils.ActionRepeatAgent(actions)

        # Rollout repeat action policy in eval. world env.
        rollout = self.eval_world_worker.rollout(eval_agent, self.world_eval_stop)
        self.eval_world_worker.reset()

        # Transform obs and actions sampled from the world env. rollout.
        prediction = rollout.sample(0).tonumpy()

        # Transform obs and actions sampled from the replay buffer.
        sample = sample.apply(
            lambda x: x[:, start + self.tau : start + self.tau + horizon]
        )
        sample = sample.apply(lambda x: np.squeeze(x, axis=0))
        sample = self.replay_to_world(sample)

        # Plot the simulated and surrogate solutions to the PDE.
        pdeplot = visual.pdeplot(
            actions=np.squeeze(sample.actions, axis=1),
            obs=np.squeeze(sample.obs, axis=1),
            opred=np.squeeze(prediction.obs, axis=1),
            rewards=sample.rewards,
            rpred=prediction.rewards,
        )
        wandb.log({"Eval. World Vec. Ep.": [wandb.Image(pdeplot)]}, commit=False)

    def update_policy(self):

        imagined = SubSeqDataset(
            data=self.world_replay.data,
            length=1,
            stride=1,
            bootstrapping=False,
            stransf=self.world_replay_to_agent
        )
        real = SubSeqDataset(
            data=self.replay.data,
            length=1,
            stride=1,
            bootstrapping=False,
            stransf=self.replay_to_agent
        )
        data = ConcatDataset((imagined, real))

        sampler = RandomSampler(
            data,
            replacement=True,
            num_samples=self.args.policy_batch_size
            * self.num_pol_updates_per_iteration,
        )

        loader = DataLoader(
            dataset=data, 
            batch_size=self.args.policy_batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=PDEDataLoader.sample_collate
        )

        for batch in loader:
            self.agent.update(batch)
            self.num_pol_updates += 1

        wandb.log({"Num. Pol. Upd.": self.num_pol_updates}, commit=False)

    def update_surrogate(self, module, trainer) -> float:
        # Extract dataset and split into training and validation datasets.
        train, val = train_test_split(
            self.replay.episodes,
            test_size=self.args.val_split_ratio,
        )

        # Determine training configuration.
        training_config = "initial" if self.iteration <= 0 else "iterations"
        training_config = self.config.training[training_config]

        datamodule = PDEDataModule(
            data=self.replay.data,
            train=train,
            val=val,
            tau=self.tau,
            stransf=self.replay_to_world,
            curriculum=self.curriculum,
            iteration=self.iteration,
            bootstrapping=True,
            **training_config
        )
        
        self.reset_trainer(trainer)

        trainer.trainer.fit(module, datamodule=datamodule)

        return trainer.trainer.logged_metrics["Val. Loss"].item()
    
    def update_delta_transform(self) -> None:
        self.undscaling.reset()
        dataset = self.replay.dataset()
        deltas = self.replay_to_world.otransf(dataset.nxtobs) - self.replay_to_world.otransf(dataset.obs)
        delta = self.env.scenario["cfg_steps"] * self.env.scenario["dt"]
        self.undscaling.update(deltas / delta)

    def summarize(self):
        summary = wandb.run.summary._as_dict()
        values = [summary[key] if key in summary else "-X-" for key in self.HEADERS]
        table = tabulate([values], headers=self.HEADERS)
        print(table)

    def end_iteration(self):
        summary = wandb.run.summary._as_dict()

        wandb.log(
            {
                "Iterations": self.iteration,
                "Num. Steps Sampled": self.num_steps_sampled
                + self.args.learning_starts,
                "Horizon": self.world.horizon,
                "World Buffer Cap.": self.imaginary_buffer_capacity,
                "World Buffer Filled": self.world_replay.ntimesteps
                / self.imaginary_buffer_capacity,
                "World Buffer Samples": self.world_replay.ntimesteps,
                "World Rollouts": self.num_world_rollouts * self.iteration,
                "Time": time.time() - summary["Start"],
            }
        )
        self.iteration += 1

    @property
    def imaginary_buffer_capacity(self):
        capacity = (
            self.args.model_buffer_store_iterations
            * self.args.model_rollouts_per_sample
            * self.samples_per_iteration
            * self.world.horizon
        )
        capacity = min(capacity, self.args.model_buffer_max_capacity)
        return capacity

    @property
    def num_world_rollouts(self):
        num = self.args.model_rollouts_per_sample * self.samples_per_iteration
        return int(num)

    @property
    def num_steps_sampled(self):
        return self.iteration * self.samples_per_iteration
