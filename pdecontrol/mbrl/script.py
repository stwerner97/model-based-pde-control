import os
import sys
import json
import argparse
import traceback
from argparse import Namespace

import torch
import wandb
import numpy as np

from pdecontrol import architectures
from pdecontrol.mbrl.mbrl import PDEModelBasedController


parser = argparse.ArgumentParser()
# ---------------- Logging & Evaluation ---------------- #
parser.add_argument("--project", type=str)
parser.add_argument("--name", type=str)
parser.add_argument("--offline", action="store_true")
parser.add_argument("--agent_eval_freq", type=int, default=50)
parser.add_argument("--num_eval_episodes", type=int, default=10)
parser.add_argument("--status_report_freq", type=int, default=5)
parser.add_argument("--logging_freq", type=int, default=10)

# ---------------- General Information ---------------- #
parser.add_argument("--total_timesteps", type=int, default=1000000)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--seed", type=int, default=0)

# ---------------- Simulation Environment & Rollouts ---------------- #
parser.add_argument("--env_id", default="KuramotoSivashinskyEnv-v0")
parser.add_argument("--cpus", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--capacity", type=int, default=1000000)
parser.add_argument("--rollout_length", type=int, default=1)

# ---------------- Model-Based Policy Optimization ---------------- #
parser.add_argument("--learning_starts", type=int, default=20000)
parser.add_argument("--policy_train_steps_per_sample", type=int, default=5)
parser.add_argument("--model_buffer_store_iterations", type=int, default=30)
parser.add_argument("--model_rollouts_per_sample", type=int, default=100)
parser.add_argument("--model_rollouts_batch_size", type=int, default=100)
parser.add_argument("--model_buffer_max_capacity", type=int, default=1000000)
parser.add_argument("--val_split_ratio", type=float, default=0.1)
parser.add_argument("--rollout_length_schedule", type=str, default="{}")

# ---------------- Surrogate World Model Training ---------------- #
parser.add_argument("--surrogate_train_freq", type=int, default=500)
parser.add_argument("--loss", type=str, default="MSELoss")
parser.add_argument("--factory", type=str)
parser.add_argument("--factory_module", type=str)
parser.add_argument("--model", type=str, default="{}")
parser.add_argument("--surrogate", type=str, default="{}")
parser.add_argument("--training", type=str, default="{}")
parser.add_argument("--curriculum", type=str, default="{}")
parser.add_argument("--trainer", type=str, default="{}")

# ---------------- Ensemble of Surrogates ---------------- #
parser.add_argument("--num_dynamics_models", type=int, default=3)
parser.add_argument("--num_elite_models", type=int, default=3)

# ---------------- Soft-Actor Critic ---------------- #
parser.add_argument("--policy", type=str, default="Gaussian")
parser.add_argument("--policy_batch_size", default=256, type=int)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--target_entropy", type=float, default=-3.0)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--target_update_interval", type=int, default=1)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--automatic_entropy_tuning", type=bool, default=False)

args = parser.parse_args()


if __name__ == "__main__":

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_SILENT"] = "true"

    wandb.init(project=args.project, config=vars(args))
    wandb.run.name = wandb.run.id if not args.name else args.name
    wandb.run.save()

    # Load the factory class using importlib.
    factory = getattr(architectures, args.factory)()

    # Use JSON to parse the CLI arguments passed as dictionaries.
    model = json.loads(args.model)
    surrogate = json.loads(args.surrogate)
    training = json.loads(args.training)
    curriculum = json.loads(args.curriculum)
    trainer = json.loads(args.trainer)

    config = {
        "factory": args.factory,
        "model": {**factory.defaults.model, **model},
        "surrogate": {**factory.defaults.surrogate, **surrogate},
        "training": {**factory.defaults.training, **training},
        "curriculum": {**factory.defaults.curriculum, **curriculum},
        "trainer": {**factory.defaults.trainer, **trainer},
        "loss": args.loss,
    }
    config = Namespace(**config)

    mbpo = PDEModelBasedController(args.env_id, factory, config, args)
    try:
        mbpo.learn()

    except Exception as e:
        print(traceback.print_exc(), file=sys.stderr)

    finally:
        wandb.finish()

