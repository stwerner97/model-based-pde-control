from typing import Dict
from abc import abstractmethod

import torch
import pytorch_lightning as pl

# Import MSELoss, so that we can import it via importlib from this file.
from torch.nn import MSELoss

from pdecontrol.surrogates.utils import Conv1dDerivative


class PhyPDELoss:
    def __init__(self, reduction: str = "none"):
        self.criterion = torch.nn.MSELoss(reduction=reduction)

    def __call__(self, augmented, *args, **kwargs):
        # After evolving, phy. targets are [û_{t+1}, ..., û_{T+1}].
        phytargets = self.phyevolve(augmented)
        phytargets = torch.cat(
            (augmented[:, -1, None, :, :], phytargets[:, :-1, :, :]), dim=1
        )

        loss = self.criterion(augmented, phytargets)
        return loss

    @abstractmethod
    def residual(self, augmented):
        pass

    @abstractmethod
    def evolve(self, augmented):
        pass


class BurgersPhyPDELoss(PhyPDELoss):
    # NOTE: PyTorch does not apply convolution in reverse mode (cross-correlation)!
    # I.e. do not reverse finite difference coefficients as in scipy.ndimage.convolve1d.
    FIRST_DERIVATIVE_SECOND_ORDER_CENTRAL = [-1 / 2, 0, 1 / 2]
    SECOND_DERIVATIVE_FOURTH_ORDER_CENTRAL = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]

    def __init__(self, dx, dt, nu, reduction: str = "none"):
        super().__init__(reduction=reduction)
        self.dx, self.dt, self.nu = dx, dt, nu

        self.grad = Conv1dDerivative(
            filter=[[self.FIRST_DERIVATIVE_SECOND_ORDER_CENTRAL]],
            resolution=(self.dx),
            kernel_size=3,
            padding=int((3 - 1) / 2),
            padding_mode="circular",
        )

        self.laplace = Conv1dDerivative(
            filter=[[self.SECOND_DERIVATIVE_FOURTH_ORDER_CENTRAL]],
            resolution=(self.dx**2),
            kernel_size=5,
            padding=int((5 - 1) / 2),
            padding_mode="circular",
        )

    def residual(self, augmented):
        len_b, len_t, len_c, len_h = augmented.shape

        # ---------------- Spatial Derivatives ---------------- #
        # Flatten [B, T, C, H] to [B * T, C, H].
        augmented = augmented.reshape(len_b * len_t, len_c, len_h)

        # Gradient_x(u) is of dimensionality [B, T, C, H] with 2nd order accuracy (first derivative).
        ux_grad = self.grad(augmented)
        ux_grad = ux_grad.reshape(len_b, len_t, len_c, len_h)

        # Laplace_x(u) is of dimensionality [B, T, C, H] with 4th order accuracy (second derivative).
        ux_laplace = self.laplace(augmented)

        # Reshape [B * T, C, H] to [B, T, C, H].
        ux_laplace = ux_laplace.reshape(len_b, len_t, len_c, len_h)

        # ---------------- PDE Residual ---------------- #
        augmented = augmented.reshape(len_b, len_t, len_c, len_h)
        return self.nu * ux_laplace - augmented * ux_grad

    def phyevolve(self, augmented):
        # Applies improved Euler scheme.
        utilde = augmented + 0.5 * self.dt * self.residual(augmented)
        return augmented + self.dt * self.residual(utilde)

    def check(self, scenario: Dict, module: pl.LightningModule):
        assert scenario["cfg_steps"] == module.surrogate.psteps
