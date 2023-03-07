from torch import nn
from munch import munchify

from pdecontrol.surrogates.surrogate import AutoRegPDESurrogate, LatentAutoRegPDESurrogate
from pdecontrol.surrogates.models.cnn import (
    ConvNet,
    ConvBlock,
    ResidualBlock,
    DeConvolutionBlock,
)
from pdecontrol.surrogates.models.fcnn import LinearBlock
from pdecontrol.surrogates.transition import (
    CNNLSTMTransitionModel,
    DelayTransitionModel,
)
from pdecontrol.surrogates.factory import PDESurrogateFactory


class KSDelayCNNSurrogateFactory(PDESurrogateFactory):

    def surrogate(self, **kwargs):
        return AutoRegPDESurrogate(**kwargs)

    def model(self, **kwargs):

        state_encoder = ConvNet(
            in_channels=1,
            out_channels=[1, 4, 8],
            blocks=[ResidualBlock, ResidualBlock, ResidualBlock],
            stride=[2, 2, 2],
            selayer=[False, False, False],
            activation=[nn.ELU, nn.ELU, nn.Tanh],
            layernorm=[nn.LayerNorm(32), nn.LayerNorm(16)]

        )

        state_decoder = ConvNet(
            in_channels=8,
            blocks=[
                DeConvolutionBlock,
                DeConvolutionBlock,
                DeConvolutionBlock,
                ConvBlock,
            ],
            out_channels=[8, 4, 1, 1],
            kernel_size=[3, 3, 3, 5],
            stride=[2, 2, 2, 1],
            padding=[1, 1, 1, 2],
            output_padding=[1, 1, 1],
            activation=[nn.ELU, nn.ELU, nn.ELU, nn.Tanh],
            layernorm=[nn.LayerNorm(16), nn.LayerNorm(32)]
        )

        action_encoder = nn.Sequential(
            LinearBlock(in_channels=1, in_size=4, out_channels=4, out_size=4, activation=nn.ELU),
            LinearBlock(in_channels=4, in_size=4, out_channels=4, out_size=8, activation=nn.Tanh),
        )

        delay = 3
        fwd_model = nn.Sequential(
            LinearBlock(in_channels=(8 + 4) * delay, in_size=8, out_channels=(8 + 4), out_size=8, activation=nn.ELU),
            LinearBlock(in_channels=(8 + 4), in_size=8, out_channels=8, out_size=8, activation=nn.ELU),
            LinearBlock(in_channels=8, in_size=8, out_channels=8, out_size=8, activation=nn.Tanh),
        )
        transition_model = DelayTransitionModel(
            schannels=8,
            ssize=8,
            achannels=4,
            asize=8,
            fwd_model=fwd_model,
            delay=delay,
        )

        return {
            "state_encoder": state_encoder,
            "state_decoder": state_decoder,
            "action_encoder": action_encoder,
            "transition_model": transition_model,
        }
