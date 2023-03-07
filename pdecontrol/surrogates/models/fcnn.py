import torch
from torch import nn


class LinearBlock(nn.Module):
    def __init__(
        self, in_channels, in_size, out_channels, out_size, activation=nn.LeakyReLU
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_size = in_size
        self.out_channels = out_channels
        self.out_size = out_size

        self.linear = nn.Linear(
            self.in_channels * self.in_size,
            self.out_channels * self.out_size,
        )
        self.activation = activation()

    def forward(self, states):
        bsize, _, _ = states.size()
        states = states.reshape(bsize, self.in_channels * self.in_size)

        states = self.linear(states)
        states = self.activation(states)
        states = states.reshape(bsize, self.out_channels, self.out_size)

        return states
