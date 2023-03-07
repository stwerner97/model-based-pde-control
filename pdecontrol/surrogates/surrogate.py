from itertools import islice
from abc import ABC, abstractmethod
from typing import List

import torch
import numpy as np
from torch import nn

from pdecontrol.mbrl.types import ModelRollout
from pdecontrol.surrogates import utils
from pdecontrol.surrogates.transition import TransitionModel
from pdegym.common.transforms import BatchTransform, Identity


class PDESurrogate(ABC, nn.Module):

    @abstractmethod
    def rollout(self, states, actions, times, targets) -> ModelRollout:
        pass


class PDEEnsemble(PDESurrogate):
    def __init__(self, modules: List, num_elites: int = None):
        super().__init__()

        self.modules = modules
        self.num_elites = num_elites
        
        if self.num_elites is None:
            self.num_elites = len(self.modules)

        self.elite_idx: List[int] = list(range(len(modules)))

    def rollout(self, states, actions, times, targets, hidden=None) -> ModelRollout:

        rollouts = []
        hidden = len(self.modules) * [None] if hidden is None else hidden

        for module, hidden in zip(self.modules, hidden):
            rollout = module.surrogate.rollout(states, actions, times, targets, hidden=hidden)
            rollouts.append(rollout)

        # Select elite modules predicting each state transition of the batch. 
        selected = np.random.choice(self.elite_idx, size=states.size(0))
        outputs = [rollouts[module_idx].outputs[idx] for idx, module_idx in enumerate(selected)]
        outputs = torch.stack(outputs, dim=0)

        # Store hidden states of all modules.
        hiddens = [output.hidden for output in rollouts]

        return ModelRollout(outputs=outputs, hidden=hiddens)

    def update_elites(self, scores: List[float]) -> None:
        idx = np.argsort(scores)[:self.num_elites]
        self.elite_idx = list(idx)
        

class AutoRegPDESurrogate(PDESurrogate):
    def __init__(
        self,
        state_encoder: nn.Module,
        state_decoder: nn.Module,
        action_encoder: nn.Module,
        transition_model: TransitionModel,
        delta: float,
        dscaling: BatchTransform = None,
        **kwargs
    ):
        super().__init__()
        
        self.delta = delta
        self.dscaling = BatchTransform(Identity()) if dscaling is None else dscaling

        self.state_encoder = utils.BatchingWrapper(state_encoder)
        self.state_decoder = utils.BatchingWrapper(state_decoder)
        self.action_encoder = utils.BatchingWrapper(action_encoder)
        self.transition_model = transition_model
    
    def rollout(self, states:torch.Tensor, actions:torch.Tensor, times:torch.Tensor, targets:torch.Tensor, hidden=None, **kwargs) -> ModelRollout:
        lstates, lactions = self.state_encoder(states), self.action_encoder(actions)

        inlast = lstates[:, 0, None, :, :]
        output = states[:, 0, None, :, :]
        
        times, targets = torch.as_tensor(times).reshape(-1), torch.as_tensor(targets).reshape(-1)

        # Determine to-be-applied action at each time point.
        timepoints = torch.arange(times[0], times[-1] + self.delta, self.delta)
        lactions = lactions[:, torch.searchsorted(times, timepoints, right=True) - 1]

        states = torch.split(states, split_size_or_sections=1, dim=1)
        lstates = torch.split(lstates, split_size_or_sections=1, dim=1)
        lactions = torch.split(lactions, split_size_or_sections=1, dim=1)

        outputs, inlatents, outlatents, outdeltas = [], [], [], []

        for state, lstate, laction in zip(states, lstates, lactions):
            inlatents.append(lstate)

            outlatent, hidden = self.transition_model.teacherforcing(states=lstate, actions=laction, hidden=hidden, **kwargs)
            outdelta = self.state_decoder(outlatent)
            output = state + self.delta * self.dscaling(outdelta)
            inlast = self.state_encoder(output).detach()

            outlatents.append(outlatent)
            outdeltas.append(outdelta)
            outputs.append(output)

        for laction in islice(lactions, len(lstates), None):
            inlatents.append(inlast)

            outlatent, hidden = self.transition_model.transition(states=inlast, actions=laction, hidden=hidden, **kwargs)    
            outdelta = self.state_decoder(outlatent)
            output = output + self.delta * self.dscaling(outdelta)
            inlast = self.state_encoder(output).detach()

            outlatents.append(outlatent)
            outdeltas.append(outdelta)
            outputs.append(output)

        inlatents = torch.cat(inlatents, dim=1)
        outlatents = torch.cat(outlatents, dim=1)
        outdeltas = torch.cat(outdeltas, dim=1)
        outputs = torch.cat(outputs, dim=1)

        targetpoints = torch.round(targets / self.delta).to(torch.long) - 1
        
        inlatents = inlatents[:, targetpoints]
        outlatents = outlatents[:, targetpoints]
        outdeltas = outdeltas[:, targetpoints]
        outputs = outputs[:, targetpoints]
        
        return ModelRollout(inlatents=inlatents, outlatents=outlatents, deltas=outdeltas, outputs=outputs, hidden=hidden)


class LatentAutoRegPDESurrogate(PDESurrogate):
    def __init__(
        self,
        state_encoder: nn.Module,
        state_decoder: nn.Module,
        action_encoder: nn.Module,
        transition_model: TransitionModel,
        delta: float,
        dscaling: BatchTransform = None,
        **kwargs
    ):
        super().__init__()
        
        self.delta = delta
        self.dscaling = BatchTransform(Identity()) if dscaling is None else dscaling

        self.state_encoder = utils.BatchingWrapper(state_encoder)
        self.state_decoder = utils.BatchingWrapper(state_decoder)
        self.action_encoder = utils.BatchingWrapper(action_encoder)
        self.transition_model = transition_model

    def rollout(self, states:torch.Tensor, actions:torch.Tensor, times:torch.Tensor, targets:torch.Tensor, hidden=None, **kwargs) -> ModelRollout:
        lstates, lactions = self.state_encoder(states), self.action_encoder(actions)

        inlatent = lstates[:, 0, None, :, :]
        
        times, targets = torch.as_tensor(times).reshape(-1), torch.as_tensor(targets).reshape(-1)

        # Determine to-be-applied action at each time point.
        timepoints = torch.arange(times[0], times[-1] + self.delta, self.delta)
        lactions = lactions[:, torch.searchsorted(times, timepoints, right=True) - 1]

        lstates = torch.split(lstates, split_size_or_sections=1, dim=1)
        lactions = torch.split(lactions, split_size_or_sections=1, dim=1)

        outputs, inlatents, outlatents = [], [], []

        for lstate, laction in zip(lstates, lactions):
            inlatents.append(lstate)

            outlatent, hidden = self.transition_model.teacherforcing(states=lstate, actions=laction, hidden=hidden, **kwargs)
            inlatent = inlatent + self.delta * outlatent
            output = self.state_decoder(inlatent)

            outlatents.append(outlatent)
            outputs.append(output)

        for laction in islice(lactions, len(lstates), None):
            inlatents.append(inlatent)

            outlatent, hidden = self.transition_model.transition(states=inlatent, actions=laction, hidden=hidden, **kwargs)    
            inlatent = inlatent + self.delta * outlatent
            output = self.state_decoder(inlatent)

            outlatents.append(outlatent)
            outputs.append(output)

        inlatents = torch.cat(inlatents, dim=1)
        outlatents = torch.cat(outlatents, dim=1)
        outputs = torch.cat(outputs, dim=1)

        outdeltas = torch.cat((states[:, 0, None], outputs), dim=1)
        outdeltas = self.dscaling.Inverse(torch.diff(outdeltas, dim=1) / self.delta) 

        targetpoints = torch.round(targets / self.delta).to(torch.long) - 1
        
        inlatents = inlatents[:, targetpoints]
        outlatents = outlatents[:, targetpoints]
        outputs = outputs[:, targetpoints]
        
        return ModelRollout(inlatents=inlatents, outlatents=outlatents, deltas=outdeltas, outputs=outputs, hidden=hidden)