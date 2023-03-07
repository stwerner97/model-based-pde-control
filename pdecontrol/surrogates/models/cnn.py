from copy import deepcopy

from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode="circular",
        bias=True,
        activation=nn.ELU,
        layernorm=None,
        **kwargs,
    ):
        super().__init__()

        self.convolution = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            padding_mode=padding_mode,
            bias=bias,
        )
        self.layernorm = layernorm

        self.activation = activation()

    def forward(self, input):
        output = self.activation(self.convolution(input))

        if self.layernorm is not None:
            output = self.layernorm(output)

        return output


class DeConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        bias=True,
        activation=nn.ELU,
        layernorm=None,
        **kwargs,
    ):
        super().__init__()
        self.deconvolution = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, bias=bias, **kwargs
        )

        self.layernorm = layernorm
        self.activation = activation()

    def forward(self, input):
        output = self.activation(self.deconvolution(input))

        if self.layernorm is not None:
            output = self.layernorm(output)
        
        return output


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding_mode="circular",
        bias=False,
        activation=nn.ELU,
        layernorm=None,
        **kwargs,
    ):
        super().__init__()


        # Residual block similar to the residual cells outlined in https://arxiv.org/pdf/2007.03898.pdf.
        self.conv3x3_l1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=(int((kernel_size - 1 ) / 2),),
            padding_mode=padding_mode,
        )
        self.conv3x3_l1_norm = deepcopy(layernorm)

        self.conv3x3_l2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=bias,
            padding=(int((kernel_size - 1 ) / 2),),
            padding_mode=padding_mode,
        )
        self.conv3x3_l2_norm = deepcopy(layernorm)

        # The skip connection applies 1x1 convolutions. It downsamples the original input if stride > 1.
        self.skip = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=bias,
            padding=(0,),
            padding_mode=padding_mode,
        )
        self.skip_norm = deepcopy(layernorm)

        self.activation = activation()

    def forward(self, x):
        # Apply 1x1 convolution along skip connection.
        identity = self.skip(x)

        # Apply convolutions along residual block.
        out = self.conv3x3_l1(x)
        out = self.activation(out)
        if self.conv3x3_l1_norm is not None:
            out = self.conv3x3_l1_norm(out)

        out = self.conv3x3_l2(out)
        out = self.activation(out)
        if self.conv3x3_l2_norm is not None:
            out = self.conv3x3_l2_norm(out)

        out = out + identity
        if self.skip_norm is not None:
            out = self.skip_norm(out)

        return out


class ConvNet(nn.Module):
    def __init__(self, in_channels, blocks, **kwargs):
        super().__init__()

        self.layers = []
        for idx, Block in enumerate(blocks):
            params = {
                param: values[idx]
                for param, values in kwargs.items()
                if len(values) > idx
            }

            block = Block(in_channels=in_channels, **params)
            in_channels = kwargs.get("out_channels")[idx]

            setattr(self, f"block_l{idx}", block)
            self.layers.append(f"block_l{idx}")

    def forward(self, inputs):
        bsize, channels, height = inputs.size()

        for name in self.layers:
            layer = getattr(self, name)
            inputs = layer(inputs)

        return inputs
