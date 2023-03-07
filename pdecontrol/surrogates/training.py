from typing import Callable
from functools import reduce

import gym
import torch
import numpy as np
import pytorch_lightning as pl

from pdecontrol.mbrl.types import ModelRollout
from pdecontrol.surrogates.surrogate import AutoRegPDESurrogate, LatentAutoRegPDESurrogate, PDESurrogate
from pdegym.common.transforms import BatchTransform, Identity, SampleTransform


class PDETrainingModule(pl.LightningModule):
    def __init__(
        self,
        surrogate: PDESurrogate,
        loss: Callable,
        tstep: float,
        delta: float,
        env: gym.Env = None,
        stransf: SampleTransform = None,
        undscaling: BatchTransform = None,
        tau: int = 5,
        tbtt: int = 10,
        lr: float = 1e-03,
        lr_gamma: float = 1.0,
        step_size: int = 25,
        **kwargs,
    ):
        super().__init__()

        self.surrogate = surrogate
        self.loss = loss
        self.tstep = tstep
        self.delta = delta
        self.env = env
        self.stransf = stransf
        self.undscaling = undscaling
        self.tau = tau
        self.tbtt = tbtt
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.step_size = step_size

        if self.stransf is None:
            self.stransf = SampleTransform()

        if isinstance(self.surrogate, AutoRegPDESurrogate):
            self.training_mode = "delta"
        
        elif isinstance(self.surrogate, LatentAutoRegPDESurrogate):
            self.training_mode = "decoded"
        else:
            raise ValueError

        if self.undscaling is None:
            self.undscaling = BatchTransform(Identity())

        assert (
            self.tbtt > self.tau
        ), "Chunk size of TBTT must be larger than warm-up length."

    def training_step(self, batch, bidx):
        states, actions, *_ = batch

        outputs = []

        # ------------ Truncated Backpropagation Through Time ------------ #
        # Split sequences into batch of length `TBTT` for Truncated Backpropagation Through Time
        schunks = list(torch.split(states, self.tbtt, dim=1))
        achunks = list(torch.split(actions, self.tbtt, dim=1))

        # Split initial chunk into warm-up and prediction portion.
        schunk, achunk = schunks.pop(0), achunks.pop(0)
        swarmup, _ = torch.split(
            schunk, (self.tau, schunk.size(1) - self.tau), dim=1
        )

        times = self.tstep * torch.arange(achunk.size(1))
        targets = self.tstep * (torch.arange(achunk.size(1)) + 1)
        output: ModelRollout = self.surrogate.rollout(states=swarmup, actions=achunk, times=times, targets=targets, hidden=None)
        outputs.append(output)

        # Detach gradient computation from hidden state & last predicted state.
        dslast = output.outputs[:, -1, None, :, :].detach()
        output.hidden = tuple(tensor.detach() for tensor in output.hidden)

        for schunk, achunk in zip(schunks, achunks):

            times = self.tstep * torch.arange(achunk.size(1))
            targets = self.tstep * (torch.arange(achunk.size(1)) + 1)
            output = self.surrogate.rollout(states=dslast, actions=achunk, times=times, targets=targets, hidden=output.hidden)
            outputs.append(output)

            # Detach gradient computation from hidden state & last predicted state.
            dslast = output.outputs[:, -1, None, :, :].detach()
            output.hidden = tuple(tensor.detach() for tensor in output.hidden)

        outdeltas = torch.cat([output.deltas for output in outputs], dim=1)[:, :-1]
        deltas = self.undscaling(torch.diff(states, dim=1) / self.delta) 

        decoded = torch.cat([output.outputs for output in outputs], dim=1)[:, :-1]
        decoded = torch.cat((states[:, 0, None, :, :], decoded), dim=1)

        if self.training_mode == "delta":
            loss = self.loss(outdeltas, deltas)
        elif self.training_mode == "decoded":
            loss = self.loss(decoded, states)

        hsteploss = loss.mean(dim=(0, 2, 3))
        loss = loss.mean()

        self.log("Train Loss", loss.detach().item(), on_step=False, on_epoch=True)
        self.log("Train Mean Delta Output", outdeltas.detach().mean(), on_step=False, on_epoch=True)
        self.log("Train Std. Delta Output", outdeltas.detach().std(), on_step=False, on_epoch=True)
        self.log("Train Mean Delta", deltas.detach().mean(), on_step=False, on_epoch=True)
        self.log("Train Std. Delta", deltas.detach().std(), on_step=False, on_epoch=True)

        outputs = torch.cat([output.outputs for output in outputs], dim=1)

        return {
            "loss": loss,
            "hsteploss": hsteploss.detach(),
            "outputs": outputs.detach(),
            "actions": actions.detach(),
            "states": states.detach(),
            "outdeltas": outdeltas.detach(),
            "deltas": deltas.detach(),
        }

    def validation_step(self, batch, bidx):
        states, actions, *_ = batch
        _, tsteps, _, _ = states.size()

        # Split sequence into warm-up and prediction phases.
        swarmup, _ = torch.split(states, (self.tau, tsteps - self.tau), dim=1)

        times = self.tstep * torch.arange(actions.size(1))
        targets = self.tstep * (torch.arange(actions.size(1)) + 1)
        outputs: ModelRollout = self.surrogate.rollout(states=swarmup, actions=actions, times=times, targets=targets, hidden=None)
        
        # The output is [u_t, ..., u_{T+1}], rather than [u_t, ..., u_T].
        # The loss expects an IC augmented state, i.e., [u_t, u_{t+1}, ...].
        decoded = torch.cat((states[:, 0, None, :, :], outputs.outputs[:, :-1, :, :]), dim=1)

        outdeltas = outputs.deltas[:, :-1]
        deltas = self.undscaling(torch.diff(states, dim=1) / self.delta)
        delta_loss = self.loss(outdeltas, deltas)
        
        self.log("Val. Delta Loss", delta_loss.detach().mean(), on_step=False, on_epoch=True)

        loss = self.loss(decoded, states).mean()
        self.log("Val. Scaled Loss", loss, on_step=False, on_epoch=True)

        # Undo applied transforms before computing the testing metrics.
        states = self.stransf.otransf.Inverse(states)
        decoded = self.stransf.otransf.Inverse(decoded)

        loss = self.loss(decoded, states)
        hsteploss = loss.detach().mean(dim=(0, 2, 3))
        loss = loss.mean()

        self.log("Val. Loss", loss, on_step=False, on_epoch=True)

        return {
            "loss": loss.detach(),
            "hsteploss": hsteploss.detach(),
            "outputs": decoded.detach(),
            "actions": actions.detach(),
            "states": states.detach(),
            "outdeltas": outdeltas.detach(),
            "deltas": deltas.detach(),
        }

    def test_step(self, batch, bidx):
        states, actions, *_ = batch
        _, tsteps, _, _ = states.size()

        # Split sequence into warm-up and prediction phases.
        swarmup, _ = torch.split(states, (self.tau, tsteps - self.tau), dim=1)

        times = self.tstep * torch.arange(actions.size(1))
        targets = self.tstep * (torch.arange(actions.size(1)) + 1)
        output: ModelRollout = self.surrogate.rollout(states=swarmup, actions=actions, times=times, targets=targets, hidden=None)
        
        # The output is [u_t, ..., u_{T+1}], rather than [u_t, ..., u_T].
        # The loss expects an IC augmented state, i.e., [u_t, u_{t+1}, ...].
        outputs = torch.cat((states[:, 0, None, :, :], output.outputs[:, :-1, :, :]), dim=1)

        # Undo applied transforms before computing the testing metrics.
        states = self.stransf.otransf.Inverse(states)
        outputs = self.stransf.otransf.Inverse(outputs)

        loss = self.loss(outputs, states).mean().numpy()

        l1_loss= torch.norm(outputs - states, p=1, dim=3).mean(dim=(0, 2)).numpy()
        l2_loss= torch.norm(outputs - states, p=2, dim=3).mean(dim=(0, 2)).numpy()

        l1_loss_scaled = (torch.norm(outputs - states, p=1, dim=3) / torch.norm(states, p=1, dim=3)).mean(dim=(0, 2)).numpy()
        l2_loss_scaled = (torch.norm(outputs - states, p=2, dim=3) / torch.norm(states, p=2, dim=3)).mean(dim=(0, 2)).numpy()
        nrmse = (torch.norm(outputs - states, p=2, dim=3)**2 / torch.norm(states, p=2, dim=3)**2).mean(dim=(0, 2)).numpy()

        # Compute reward estimates given predicted states.
        bsize, steps, achannels, aheight = actions.shape
        bsize, steps, schannels, sheight = states.shape
        actions = actions.reshape(bsize * steps, achannels, aheight)
        states = states.reshape(bsize * steps, schannels, sheight)
        outputs = outputs.reshape(bsize * steps, schannels, sheight)

        phi = self.stransf.atransf.Inverse(actions)
        phi = BatchTransform(self.env.forcing)(phi)

        rews = torch.stack([self.env.reward_func(s, p) for s, p in zip(states, phi)], dim=0)
        pred_rews = torch.stack([self.env.reward_func(o, p) for o, p in zip(outputs, phi)], dim=0)

        rews = rews.reshape(bsize, steps)
        pred_rews = pred_rews.reshape(bsize, steps)

        l1_loss_rews= torch.norm(rews - pred_rews, p=1, dim=0).numpy()
        l2_loss_rews= torch.norm(rews - pred_rews, p=2, dim=0).numpy()

        l1_loss_scaled_rews = (torch.norm(rews - pred_rews, p=1, dim=0) / torch.norm(rews, p=1, dim=0)).numpy()
        l2_loss_scaled_rews = (torch.norm(rews - pred_rews, p=2, dim=0) / torch.norm(rews, p=2, dim=0)).numpy()
        nrmse_rews = (torch.norm(rews - pred_rews, p=2, dim=0)**2 / torch.norm(rews, p=2, dim=0)**2).numpy()

        # Compute spatial derivatives of states and predicted states.
        derivs = [self.env.rhs(s, p) for s, p in zip(states.numpy(), phi.numpy())]
        derivs = torch.as_tensor(np.asarray([list(deriv) for _, deriv in derivs]))
        _, nderivs, channels, height = derivs.shape
        derivs = derivs.reshape(bsize, steps, nderivs, channels, height)

        pred_derivs = [self.env.rhs(o, p) for o, p in zip(outputs.numpy(), phi.numpy())]
        pred_derivs = torch.as_tensor(np.asarray([list(deriv) for _, deriv in pred_derivs]))
        _, nderivs, channels, height = pred_derivs.shape
        pred_derivs = pred_derivs.reshape(bsize, steps, nderivs, channels, height)

        l1_loss_derivs = torch.norm(derivs - pred_derivs, p=1, dim=4).mean(dim=(0, 3))
        l2_loss_derivs = torch.norm(derivs - pred_derivs, p=2, dim=4).mean(dim=(0, 3))

        l1_loss_scaled_derivs = (torch.norm(derivs - pred_derivs, p=1, dim=4) / torch.norm(derivs, p=1, dim=4)).mean(dim=(0, 3))
        l2_loss_scaled_derivs = (torch.norm(derivs - pred_derivs, p=2, dim=4) / torch.norm(derivs, p=2, dim=4)).mean(dim=(0, 3))
        nrms_derivs = (torch.norm(derivs - pred_derivs, p=2, dim=4)**2 / torch.norm(derivs, p=2, dim=4)**2).mean(dim=(0, 3))

        names = ("l1_loss_derivs", "l2_loss_derivs", "l1_loss_scaled_derivs", "l2_loss_scaled_derivs", "nrms_derivs")
        data = (l1_loss_derivs, l2_loss_derivs, l1_loss_scaled_derivs, l2_loss_scaled_derivs, nrms_derivs)
        data = [{f"{name}-derivative-{idx}": values for idx, values in enumerate(drvloss.T)} for name, drvloss in zip(names, data)]
        data = reduce(lambda a, b: {**a, **b}, data)

        actions = actions.reshape(bsize, steps, achannels, aheight)
        states = states.reshape(bsize, steps, schannels, sheight)
        outputs = outputs.reshape(bsize, steps, schannels, sheight)

        data = {
            "states": states.numpy(),
            "outputs": outputs.numpy(),
            "actions": actions.numpy(),
            "MSE": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l1_loss_scaled": l1_loss_scaled,
            "l2_loss_scaled": l2_loss_scaled,
            "nrmse": nrmse,
            "l1_loss_rews": l1_loss_rews,
            "l2_loss_rews": l2_loss_rews,
            "l1_loss_scaled_rews": l1_loss_scaled_rews,
            "l2_loss_scaled_rews": l2_loss_scaled_rews,
            "nrmse_rews": nrmse_rews,
            **data
        }
        return data

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.lr_gamma
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
