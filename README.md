# Towards sample-efficient Reinforcement Learning with data-driven surrogate models and physical constraints

This repository holds the implementation of a model-based deep reinforcement learning algorithm for the control of partial differential equations. 

## Setup
The dependencies can be installed using the provided ``Dockerfile``. Our implementation logs intermediate evaluation results to ``Weights & Biases``, so an API key must be provided.

## Execution
The ``pdecontrol/mbrl/script.py`` file serves as an entrypoint to running experiments for our control algorithm. It can be used to specify the configuration of the agent, for example, as follows

````python
python pdecontrol/mbrl/script.py
    --project <your wandb project>
    --name <your wandb run>
    --env_id KuramotoSivashinskyEnv-v0
    --factory <name of dynamics model registered in pdecontrol/architectures/__init__.py>
    --training "{\"tau\": 5, \"initial\": {\"tbtt\": 10, \"patience\": 10, \"batch_size\": 64}, \"iterations\": {\"tbtt\": 10, \"patience\": 5, \"batch_size\": 64}}"
    --trainer "{\"initial\": {\"min_steps\":250, \"max_steps\": 2000}, \"iterations\": {\"min_steps\":50, \"max_steps\": 250}}"
    --curriculum "{\"scheduler\": \"LinearScheduler\", \"steptype\": \"iteration\", \"start\": 0, \"stop\": 10, \"vmin\": 15, \"vmax\": 15}"
    --loss MSELoss
    --learning_starts 5000
    --rollout_length_schedule "{\"scheduler\": \"LinearScheduler\", \"steptype\": \"iteration\", \"start\": 0, \"stop\": 200, \"vmin\": 3, \"vmax\": 7}" --policy_train_steps_per_sample 10
    --surrogate_train_freq 500
````
