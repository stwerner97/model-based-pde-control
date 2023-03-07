from abc import abstractmethod

import torch
from torch import nn


class TransitionModel(nn.Module):
    def __init__(
        self,
        schannels: int,
        ssize: int,
        achannels: int,
        asize: int,
        dtype=torch.FloatTensor,
        **kwargs
    ):
        super().__init__()

        self.dtype = dtype
        self.schannels = schannels
        self.ssize = ssize
        self.achannels = achannels
        self.asize = asize

    @abstractmethod
    def teacherforcing(self, states, actions, hidden=None, **kwargs):
        pass

    @abstractmethod
    def transition(self, states, actions, hidden=None, **kwargs):
        pass


class LSTMTransitionModel(TransitionModel):
    def __init__(
        self,
        schannels: int,
        ssize: int,
        achannels: int,
        asize: int,
        dtype=torch.FloatTensor,
    ):
        super().__init__(schannels, ssize, achannels, asize, dtype)

        # Setup LSTM model that expects flattened inputs.
        self.lstm = nn.LSTM(
            self.achannels * self.asize, self.schannels * self.ssize, batch_first=True
        )

        # Initialize parameters for *NOT*-learnable initial hidden & cell state.
        self.H0 = nn.Parameter(
            torch.zeros(self.schannels * self.ssize).type(self.dtype),
            requires_grad=False,
        )
        self.C0 = nn.Parameter(
            torch.zeros(self.schannels * self.ssize).type(self.dtype),
            requires_grad=False,
        )

    def teacherforcing(self, states, actions, hidden=None, **kwargs):
        bsize, ssteps, _, _ = states.size()

        if hidden is None:
            H = self.H0.repeat(bsize, 1).unsqueeze(0)
            C = self.C0.repeat(bsize, 1).unsqueeze(0)
            hidden = (H, C)

        H, C = hidden

        # Flatten passed latent states & actions.
        states = states.reshape(bsize, ssteps, self.schannels * self.ssize)
        actions = actions.reshape(bsize, ssteps, self.achannels * self.asize)

        # Hidden state is expected to be of shape [1, BSIZE, SCHANNELS * SSIZE].
        states = torch.swapaxes(states, 0, 1)

        outputs = []

        # Warm-Up phase uses teacher forcing (i.e. replaces H).
        for widx in range(ssteps):
            inputs = actions[:, widx, None, :]
            H = states[widx, None, :, :]
            output, (H, C) = self.lstm(inputs, (H, C))
            outputs.append(output)

        outputs = torch.cat((outputs), dim=1)
        outputs = outputs.reshape(bsize, ssteps, self.schannels, self.ssize)

        return outputs, (H, C)

    def transition(self, states, actions, hidden=None, **kwargs):
        bsize, asteps, _, _ = actions.size()

        if hidden is None:
            H = self.H0.repeat(bsize, 1).unsqueeze(0)
            C = self.C0.repeat(bsize, 1).unsqueeze(0)
            hidden = (H, C)

        H, C = hidden

        # Flatten passed latent states & actions.
        actions = actions.reshape(bsize, asteps, self.achannels * self.asize)

        # Use LSTM in generation mode given the initial hidden state H & cell state C.
        outputs, hidden = self.lstm(actions, hidden)

        outputs = outputs.reshape(bsize, asteps, self.schannels, self.ssize)

        return outputs, hidden


class CNNLSTMCell(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        height,
        kernel_size=3,
        stride=1,
        bias=True,
        padding_mode="circular",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.padding = int((self.kernel_size - 1) / 2)

        self.Wxi = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=self.bias,
            padding_mode=padding_mode,
        )

        self.Whi = nn.Conv1d(
            self.out_channels,
            self.out_channels,
            self.kernel_size,
            1,
            padding=self.padding,
            bias=False,
            padding_mode=padding_mode,
        )

        self.Wxf = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=True,
            padding_mode=padding_mode,
        )

        self.Whf = nn.Conv1d(
            self.out_channels,
            self.out_channels,
            self.kernel_size,
            1,
            padding=1,
            bias=False,
            padding_mode=padding_mode,
        )

        self.Wxc = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=True,
            padding_mode=padding_mode,
        )

        self.Whc = nn.Conv1d(
            self.out_channels,
            self.out_channels,
            self.kernel_size,
            1,
            padding=self.padding,
            bias=False,
            padding_mode=padding_mode,
        )

        self.Wxo = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=True,
            padding_mode=padding_mode,
        )

        self.Who = nn.Conv1d(
            self.out_channels,
            self.out_channels,
            self.kernel_size,
            1,
            padding=self.padding,
            bias=False,
            padding_mode=padding_mode,
        )

        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc


class CNNLSTMTransitionModel(TransitionModel):
    def __init__(
        self,
        schannels,
        ssize,
        achannels,
        asize,
        kernel_size=3,
        stride=1,
        bias=True,
        Cell=CNNLSTMCell,
        dtype=torch.FloatTensor,
    ):
        super().__init__(schannels, ssize, achannels, asize, dtype)

        self.cnnlstmcell = Cell(
            in_channels=achannels,
            out_channels=schannels,
            height=ssize,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

        # Initialize parameters for *NOT*-learnable initial input & cell state.
        self.H0 = nn.Parameter(
            torch.zeros(schannels, ssize).type(self.dtype), requires_grad=False
        )
        self.C0 = nn.Parameter(
            torch.zeros(schannels, ssize).type(self.dtype), requires_grad=False
        )

    def teacherforcing(self, states, actions, hidden=None, **kwargs):
        bsize, ssteps, _, _ = states.size()
        _, asteps, _, _ = actions.size()
        assert ssteps == asteps

        if hidden is None:
            H = self.H0.repeat(bsize, 1, 1)
            C = self.C0.repeat(bsize, 1, 1)
            hidden = (H, C)

        outputs = []
        H, C = hidden

        for widx in range(ssteps):
            # Warm-Up phase uses teacher forcing (i.e. replaces H).
            H = states[:, widx, :, :]

            H, C = self.cnnlstmcell(actions[:, widx, :], H, C)
            outputs.append(H)

        outputs = torch.stack((outputs), dim=1)

        return outputs, (H, C)

    def transition(self, states, actions, hidden, **kwargs):
        bsize, asteps, _, _ = actions.size()

        outputs = []
        H, C = hidden

        for idx in range(asteps):
            H, C = self.cnnlstmcell(actions[:, idx, :, :], H, C)
            outputs.append(H)

        outputs = torch.stack(outputs, dim=1)
        return outputs, (H, C)


class DelayTransitionModel(TransitionModel):
    def __init__(
        self,
        schannels: int,
        ssize: int,
        achannels: int,
        asize: int,
        fwd_model: nn.Module,
        delay: int,
        dtype=torch.FloatTensor,
    ):
        super().__init__(
            schannels=schannels,
            ssize=ssize,
            achannels=achannels,
            asize=asize,
            dtype=dtype,
        )

        self.delay = delay
        self.fwd_model = fwd_model

    def forward(self, scontext, acontext):
        bsize, ssteps, schannels, ssize = scontext.shape
        _, asteps, achannels, asize = acontext.shape
        assert ssteps == self.delay
        assert asteps == self.delay
        assert ssize == asize

        augmented = torch.cat((scontext, acontext), dim=2)
        augmented = augmented.reshape(bsize, self.delay * (schannels + achannels), ssize)
        nxtstate = self.fwd_model(augmented)
        nxtstate = nxtstate.reshape(bsize, 1, self.schannels, self.ssize)
        return nxtstate

    def teacherforcing(self, states, actions, hidden=None, **kwargs):
        bsize, ssteps, _, _ = states.size()
        _, asteps, _, _ = actions.size()
        assert ssteps == asteps

        if hidden is None:
            scontext = torch.zeros(bsize, self.delay, self.schannels, self.ssize, dtype=torch.float32)
            acontext = torch.zeros(bsize, self.delay, self.achannels, self.asize, dtype=torch.float32)
            hidden = (scontext, acontext)

        (scontext, acontext) = hidden

        outputs = []

        for idx in range(ssteps):
            scontext[:, 0] = states[:, idx]
            scontext = torch.roll(scontext, shifts=-1, dims=1)

            acontext[:, 0] = actions[:, idx]
            acontext = torch.roll(acontext, shifts=-1, dims=1)

            spredicted = self(scontext, acontext)
            outputs.append(spredicted)

        outputs = torch.cat(outputs, dim=1)

        return outputs, (scontext, acontext)

    def transition(self, states, actions, hidden=None, **kwargs):
        bsize, ssteps, _, _ = states.size()
        _, asteps, _, _ = actions.size()
        assert ssteps == 1
        assert asteps == 1

        if hidden is None:
            scontext = torch.zeros(bsize, self.delay, self.schannels, self.ssize, dtype=torch.float32)
            acontext = torch.zeros(bsize, self.delay, self.achannels, self.asize, dtype=torch.float32)
            hidden = (scontext, acontext)

        (scontext, acontext) = hidden

        scontext[:, 0] = states[:, 0].clone().detach()
        scontext = torch.roll(scontext, shifts=-1, dims=1)

        acontext[:, 0] = actions[:, 0]
        acontext = torch.roll(acontext, shifts=-1, dims=1)

        output = self(scontext, acontext)
        return output, (scontext, acontext)
