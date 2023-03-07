from torch import nn

from pdecontrol.surrogates.surrogate import AutoRegPDESurrogate
from pdecontrol.surrogates.models.fcnn import LinearBlock
from pdecontrol.surrogates.models import cnn as CNN
from pdecontrol.surrogates.transition import LSTMTransitionModel, CNNLSTMTransitionModel
from pdecontrol.surrogates.factory import PDESurrogateFactory


class KSAutoRegFullyConnectedLSTM(PDESurrogateFactory):
    """ Spatial & Temporal Locality Ablation. """

    def surrogate(self, **kwargs):
        return AutoRegPDESurrogate(**kwargs)

    def model(self, **kwargs):
        state_encoder = nn.Sequential(
            LinearBlock(in_channels=1, in_size=64, out_channels=1, out_size=32, activation=nn.SiLU),
            LinearBlock(in_channels=1, in_size=32, out_channels=1, out_size=16, activation=nn.SiLU)
        )

        state_decoder = nn.Sequential(
            LinearBlock(in_channels=1, in_size=16, out_channels=1, out_size=32, activation=nn.SiLU),
            LinearBlock(in_channels=1, in_size=32, out_channels=1, out_size=64, activation=nn.Tanh)
        )

        action_encoder = nn.Identity()

        transition_model = LSTMTransitionModel(
            schannels =1,
            ssize = 16,
            achannels=1,
            asize=4
        )

        return {
            "state_encoder": state_encoder,
            "state_decoder": state_decoder,
            "action_encoder": action_encoder,
            "transition_model": transition_model,
        }


class KSAutoRegConvolutionalLSTM(PDESurrogateFactory):

    def surrogate(self, **kwargs):
        return AutoRegPDESurrogate(**kwargs)

    def model(self, **kwargs):

        state_encoder = CNN.ConvNet(
            in_channels=1,
            blocks=[
                CNN.ResidualBlock, CNN.ResidualBlock, CNN.ResidualBlock
            ],
            out_channels=[8, 16, 16],
            kernel_size=[3, 3, 3],
            stride=[2, 2, 1],
            activation=[nn.SiLU, nn.SiLU, nn.SiLU],
            layernorm=[nn.LayerNorm(32), nn.LayerNorm(16), nn.LayerNorm(16)]
        )

        action_encoder = CNN.ConvNet(
            in_channels=1,
            blocks=[
                CNN.ResidualBlock, CNN.ResidualBlock, CNN.ResidualBlock
            ],
            out_channels=[2, 4, 4],
            kernel_size=[3, 3, 3],
            stride=[2, 2, 1],
            activation=[nn.SiLU, nn.SiLU, nn.SiLU],
            layernorm=[nn.LayerNorm(32), nn.LayerNorm(16), nn.LayerNorm(16)]
        )

        transition_model = CNNLSTMTransitionModel(
            schannels=16, ssize=16, achannels=4, asize=16
        )

        state_decoder = CNN.ConvNet(
            in_channels=16,
            blocks=[
                CNN.DeConvolutionBlock,
                CNN.DeConvolutionBlock,
                CNN.ConvBlock,
                CNN.ConvBlock,
            ],
            out_channels=[16, 8, 1, 1],
            kernel_size=[3, 3, 7, 5],
            stride=[2, 2, 1, 1],
            padding=[1, 1, 3, 2],
            output_padding=[1, 1],
            activation=[nn.SiLU, nn.SiLU, nn.SiLU, nn.Identity],
            layernorm=[nn.LayerNorm(32), nn.LayerNorm(64), nn.LayerNorm(64)]
        )

        return {
            "state_encoder": state_encoder,
            "state_decoder": state_decoder,
            "action_encoder": action_encoder,
            "transition_model": transition_model,
        }
