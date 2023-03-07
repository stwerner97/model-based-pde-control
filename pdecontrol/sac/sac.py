from argparse import Namespace

import wandb
import torch
import torch.nn.functional as F
from torch.optim import Adam

from pdecontrol.sac.utils import soft_update, hard_update
from pdecontrol.sac.policies import GaussianPolicy, QNetwork


class SAC(object):
    def __init__(self, observation_space, action_space, config: Namespace):

        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha

        self.policy_type = config.policy
        self.target_update_interval = config.target_update_interval
        self.automatic_entropy_tuning = config.automatic_entropy_tuning

        self.device = torch.device("cuda" if config.cuda else "cpu")

        ochannels, oheight = observation_space.shape
        achannels, aheight = action_space.shape

        self.critic = QNetwork(
            ochannels, oheight, achannels, aheight, config.hidden_size
        ).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=config.lr)

        self.critic_target = QNetwork(
            ochannels, oheight, achannels, aheight, config.hidden_size
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.updates = 0

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(
                torch.Tensor(action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=config.lr)

        self.policy = GaussianPolicy(
            ochannels, oheight, achannels, aheight, config.hidden_size, action_space
        ).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=config.lr)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()

    def update(self, batch):
        obs, actions, nxtobs, rewards, terminated, truncated, steps = batch
        obs, actions, nxtobs = tuple(
            map(lambda tensor: tensor.squeeze(1), (obs, actions, nxtobs))
        )

        wandb.log({"Pol. Rew. Mean": torch.mean(rewards)}, commit=False)

        terminated = terminated.type(torch.FloatTensor)
        truncated = truncated.type(torch.FloatTensor)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask_batch = 1.0 - terminated
        assert torch.all(mask_batch >= 1.0).item()
        assert torch.all(mask_batch <= 1.0).item()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(nxtobs)
            qf1_next_target, qf2_next_target = self.critic_target(
                nxtobs, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = rewards + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(
            obs, actions
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(obs)

        qf1_pi, qf2_pi = self.critic(obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        self.updates += 1

        wandb.log(
            {
                "SAC/Qloss": qf_loss.item(),
                "SAC/PolicyLoss": policy_loss.item(),
                "SAC/entropy_loss": alpha_loss.item(),
                "SAC/alpha_loss": alpha_tlogs.item(),
            }
        )
