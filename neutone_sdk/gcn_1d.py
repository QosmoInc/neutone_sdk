import logging
import os
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class TFiLM(nn.Module):
    """Temporal Feature-wise Linear Modulation (TFiLM) layer.

    Parameters:
        n_channels (int): Number of channels in the input signal.
        cond_dim (int): Dimensionality of the conditional input.
        tfilm_block_size (int): Size of the temporal blocks.
        rnn_type (str, optional): Type of RNN to use for the modulation.

    Returns:
        Tensor: The output of the TFiLM layer.
    """

    def __init__(
        self,
        n_channels: int,
        cond_dim: int,
        tfilm_block_size: int,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.nchannels = n_channels
        self.cond_dim = cond_dim
        self.tfilm_block_size = tfilm_block_size
        self.num_layers = 1
        self.first_run = True
        self.hidden_state = (
            torch.Tensor(0),
            torch.Tensor(0),
        )  # (hidden_state, cell_state)

        self.maxpool = torch.nn.MaxPool1d(
            kernel_size=tfilm_block_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )

        rnn_types = {"lstm": torch.nn.LSTM, "gru": torch.nn.GRU}

        try:
            RNN = rnn_types[rnn_type.lower()]
            self.rnn = RNN(
                input_size=n_channels + cond_dim,
                hidden_size=n_channels,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=False,
            )
        except KeyError:
            raise ValueError(f"Invalid rnn_type. Use 'lstm' or 'gru'. Got {rnn_type}")

    def forward(self, x: Tensor , cond: Optional[Tensor] = None) -> Tensor:
        x_in_shape = x.shape  # (batch_size, n_channels, samples)

        # Pad input to be divisible by tfilm_block_size
        if (x_in_shape[2] % self.tfilm_block_size) != 0:
            padding = torch.zeros(
                x_in_shape[0],
                x_in_shape[1],
                self.tfilm_block_size - (x_in_shape[2] % self.tfilm_block_size),
            )
            x = torch.cat((x, padding), dim=-1)

        x_shape = x.shape
        n_steps = int(x_shape[-1] / self.tfilm_block_size)

        x_down = self.maxpool(x)  # (batch_size, n_channels, n_steps)

        if cond is not None:
            cond_up = cond.unsqueeze(-1)
            cond_up = cond_up.repeat(1, 1, n_steps)  # (batch_size, cond_dim, n_steps)
            x_down = torch.cat(
                (x_down, cond_up), dim=1
            )  # (batch_size, n_channels + cond_dim, n_steps)

        # Put shape to (n_steps, batch_size, n_channels + cond_dim)
        x_down = x_down.permute(2, 0, 1)

        # Modulation
        if self.first_run:  # Reset hidden state
            x_norm, self.hidden_state = self.rnn(x_down, None)
            self.first_run = False
        else:
            x_norm, self.hidden_state = self.rnn(x_down, self.hidden_state)

        # Put shape back to (batch_size, n_channels, length)
        x_norm = x_norm.permute(1, 2, 0)

        # Reshape input and modulation sequence into blocks
        x_in = torch.reshape(
            x, shape=(-1, self.nchannels, n_steps, self.tfilm_block_size)
        )
        x_norm = torch.reshape(x_norm, shape=(-1, self.nchannels, n_steps, 1))

        x_out = x_norm * x_in

        # Return to the original padded input shape
        x_out = torch.reshape(x_out, shape=(x_shape))

        x_out = x_out[..., : x_in_shape[2]]  # Remove padding

        return x_out

    def reset_state(self) -> None:
        self.first_run = True


class Conv1dCausal(nn.Module):
    """Causal 1D convolutional layer
    ensures outputs depend only on current and past inputs.

    Parameters:
        in_channels (int): Number of channels in the input signal.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        dilation (int, optional): Spacing between kernel elements.
        bias (bool, optional): If True, adds a learnable bias to the output.

    Returns:
        Tensor: The output of the causal 1D convolutional layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.padding = (
            kernel_size - 1
        ) * dilation  # input_len == output_len when stride=1
        self.in_channels = in_channels
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            (kernel_size,),
            (stride,),
            padding=0,
            dilation=(dilation,),
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.padding, 0))  # standard zero padding
        x = self.conv(x)
        return x


class GatedAF(nn.Module):
    """Gated activation function
    applies a tanh activation to one half of the input
    and a sigmoid activation to the other half, and then multiplies them element-wise.

    Returns:
        Tensor: The output of the gated activation function.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x_tanh, x_sigmoid = x.chunk(2, dim=1)  # Split the output into two halves

        x_tanh = torch.tanh(x_tanh)  # Apply tanh activation
        x_sigmoid = torch.sigmoid(x_sigmoid)  # Apply sigmoid activation

        # Element-wise multiplication of tanh and sigmoid activations
        x = x_tanh * x_sigmoid
        return x


class GCN1DBlock(nn.Module):
    """Single block of a Gated Convolutional Network (GCN) with conditional modulation.

    Parameters:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel.
        dilation (int, optional): Dilation rate for dilated convolutions.
        stride (int, optional): Stride for the convolution.
        cond_dim (int, optional): Dimensionality of the conditional input for FiLM.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        cond_dim: int = 0,
        rnn_type: str = "lstm",
        tfilm_block_size: int = 128,
        use_bias_in_conv: bool = False,
    ) -> None:
        super().__init__()

        self.conv = Conv1dCausal(
            in_channels=in_ch,
            out_channels=out_ch * 2,  # adapt for the Gated Activation Function
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias_in_conv,
        )

        self.tfilm = None
        if cond_dim > 0:
            self.tfilm = TFiLM(
                n_channels=out_ch * 2,
                cond_dim=cond_dim,
                tfilm_block_size=tfilm_block_size,
                rnn_type=rnn_type,
            )

        self.gated_activation = GatedAF()

        self.res = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=(1,), bias=False
        )

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        x_in = x
        x = self.conv(x)  # Apply causal convolution
        if (
            cond is not None and self.tfilm is not None
        ):  # Apply FiLM if conditional input is given
            x = self.tfilm(x, cond)
        # Apply gated activation function
        x = self.gated_activation(x)
        # Apply residual convolution and add to output
        x_res = self.res(x_in)
        x = x + x_res
        return x


class GCN1D(nn.Module):
    """Gated Convolutional Network (GCN) model, re-implemented from the paper:
    https://arxiv.org/abs/2211.00497

    Parameters:
        in_ch (int, optional): Number of input channels.
        out_ch (int, optional): Number of output channels.
        n_blocks (int, optional): Number of GCN blocks.
        n_channels (int, optional): Number of channels in the GCN blocks.
        dilation_growth (int, optional): Growth rate for dilation in the GCN blocks.
        kernel_size (int, optional): Size of the convolution kernel.
        cond_dim (int, optional): Dimensionality of the conditional input for FiLM.

    Returns:
        Tensor: The output of the GCN model.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        n_blocks: int = 10,
        n_channels: int = 64,
        dil_growth: int = 4,
        kernel_size: int = 13,
        cond_dim: int = 0,
        tfilm_block_size: int = 128,
        rnn_type: str = "lstm",
        use_act: bool = True,
        use_bias_in_conv: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.dil_growth = dil_growth
        self.n_blocks = n_blocks
        self.cond_dim = cond_dim
        self.use_act = use_act
        self.use_bias_in_conv = use_bias_in_conv

        # Compute convolution channels and dilations
        self.channels = [n_channels] * n_blocks
        self.dilations = [dil_growth**idx for idx in range(n_blocks)]

        # Blocks number is given by the number of elements in the channels list
        self.n_blocks = len(self.channels)
        assert len(self.dilations) == self.n_blocks

        # Create a list of strides
        self.strides = [1] * self.n_blocks

        # Create a list of GCN blocks
        self.blocks = nn.ModuleList()
        block_out_ch = 0

        for idx, (curr_out_ch, dil, stride) in enumerate(
            zip(self.channels, self.dilations, self.strides)
        ):
            block_out_ch = curr_out_ch
            if idx == 0:
                block_in_ch = in_ch
            else:
                block_in_ch = block_out_ch

            self.blocks.append(
                GCN1DBlock(
                    block_in_ch,
                    block_out_ch,
                    self.kernel_size,
                    dilation=dil,
                    stride=stride,
                    cond_dim=cond_dim,
                    tfilm_block_size=tfilm_block_size,
                    rnn_type=rnn_type,
                    use_bias_in_conv=use_bias_in_conv,
                )
            )

        # Output layer
        self.out_net = nn.Conv1d(
            self.channels[-1], out_ch, kernel_size=(1,), stride=(1,), bias=False
        )

        # Activation function
        self.act = nn.Tanh()

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        if cond is not None:
            assert cond.ndim == 2  # (batch_size, cond_dim)
        for block in self.blocks:  # Apply GCN blocks
            x = block(x, cond)
        x = self.out_net(x)  # Apply output layer

        if self.act is not None:
            x = self.act(x)  # Apply tanh activation function
        return x

    def calc_receptive_field(self) -> int:
        """Calculate the receptive field of the model.
        The receptive field is the number of input samples that affect the output of a block.

        The receptive field of the model is the sum of the receptive fields of all layers:
        RF = 1 + \sum_{i=1}^{n}(kernel\_size_i - 1) \cdot dilation_i

        i is the layer index, n is the number of layers.

        Returns:
            int: The receptive field of the model.
        """
        assert all(_ == 1 for _ in self.strides)  # TODO(cm): add support for dsTCN
        assert self.dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
        rf = self.kernel_size
        for dil in self.dilations[1:]:
            rf = rf + ((self.kernel_size - 1) * dil)
        return rf

