import logging
import os
from typing import Optional, List, Union, Tuple

import torch as tr
from torch import Tensor
from torch import nn

from neutone_sdk.conv import Conv1dGeneral

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class FiLM(nn.Module):
    def __init__(self,
                 cond_dim: int,  # Dim of conditioning input
                 num_features: int,  # Dim of the conv channel
                 use_bn: bool) -> None:
        super().__init__()
        self.num_features = num_features
        self.use_bn = use_bn
        self.bn = None
        if self.use_bn:
            self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = nn.Linear(cond_dim, 2 * num_features)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        cond = self.adaptor(cond)
        g, b = tr.chunk(cond, 2, dim=-1)
        g = g.unsqueeze(-1)
        b = b.unsqueeze(-1)
        if self.bn is not None:
            x = self.bn(x)  # Apply batchnorm without affine
        x = (x * g) + b  # Then apply conditional affine
        return x


class TCNBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Union[str, int, Tuple[int]] = "same",
                 dilation: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 causal: bool = True,
                 cached: bool = False,
                 use_dynamic_bs: bool = True,
                 batch_size: int = 1,
                 use_ln: bool = False,
                 temporal_dim: Optional[int] = None,
                 use_act: bool = True,
                 act_name: str = "prelu",
                 use_res: bool = True,
                 cond_dim: int = 0,
                 use_film_bn: bool = True,  # TODO(cm): check if this should be false
                 debug_mode: bool = True) -> None:
        super().__init__()
        self.use_ln = use_ln
        self.temporal_dim = temporal_dim
        self.use_act = use_act
        self.act_name = act_name
        self.use_res = use_res
        self.cond_dim = cond_dim
        self.use_film_bn = use_film_bn
        self.debug_mode = debug_mode

        self.ln = None
        if use_ln:
            assert temporal_dim is not None and temporal_dim > 0
            self.ln = nn.LayerNorm(
                [in_channels, temporal_dim], elementwise_affine=False)

        self.act = None
        if use_act:
            self.act = self.get_activation(act_name, out_channels)

        self.conv = Conv1dGeneral(in_channels,
                                  out_channels,
                                  kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias,
                                  padding_mode=padding_mode,
                                  causal=causal,
                                  cached=cached,
                                  use_dynamic_bs=use_dynamic_bs,
                                  batch_size=batch_size,
                                  debug_mode=debug_mode)
        self.res = None
        if use_res:
            self.res = nn.Conv1d(in_channels,
                                 out_channels,
                                 kernel_size=(1,),
                                 stride=(stride,),
                                 bias=False)

        self.film = None
        if cond_dim > 0:
            self.film = FiLM(cond_dim, out_channels, use_bn=use_film_bn)

    @tr.jit.export
    def is_conditional(self) -> bool:
        """Returns True if the TCN block is conditional, False otherwise."""
        return self.cond_dim > 0

    @tr.jit.export
    def is_cached(self) -> bool:
        """Returns True if the TCN block is cached, False otherwise."""
        return self.conv.is_cached()

    @tr.jit.export
    def set_cached(self, cached: bool) -> None:
        """
        Sets the TCN block to cached or not cached mode and resets its state.

        Args:
            cached: If True, the TCN block is cached. If False, it is not cached.
        """
        self.conv.set_cached(cached)

    @tr.jit.export
    def reset(self, batch_size: Optional[int] = None) -> None:
        """
        Resets the TCN block's state. If batch_size is provided, the cached padding
        will be resized to match the new batch size.

        Args:
            batch_size: If provided, the cached padding will be resized to match the new
                        batch size.
        """
        self.conv.reset(batch_size)

    @tr.jit.export
    def get_delay_samples(self) -> int:
        """
        Returns the number of samples that the TCN block delays the output by. This
        should always be 0 when the TCN block is causal. This is ill-defined when not
        in cached mode since the output number of samples can be different than the
        input number of samples, so this would typically only be used in cached mode.
        """
        return self.conv.get_delay_samples()

    def prepare_for_inference(self) -> None:
        """
        Prepares the TCN block for inference by disabling debug mode and ensuring the
        TCN block is in cached mode.
        """
        self.debug_mode = False
        self.conv.prepare_for_inference()
        self.eval()

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        if self.debug_mode:
            assert x.ndim == 3  # (batch_size, in_ch, samples)
        x_in = x
        if self.ln is not None:
            if self.debug_mode:
                assert x.size(1) == self.in_ch
                assert x.size(2) == self.temporal_dim
            x = self.ln(x)
        x = self.conv(x)
        if self.film is not None:
            if self.debug_mode:
                assert cond is not None
            if cond is not None:  # This if statement is needed for TorchScript
                x = self.film(x, cond)
        if self.act is not None:
            x = self.act(x)
        if self.res is not None:
            res = self.res(x_in)
            right_offset = self.get_delay_samples()
            x_res = Conv1dGeneral.right_offset_crop(res, x.size(-1), right_offset)
            x += x_res
        return x

    @staticmethod
    def get_activation(act_name: str, out_ch: Optional[int] = None) -> nn.Module:
        """
        Most of the code and experimental results in this method are from
        https://github.com/csteinmetz1/ronn

        Given an activation name string, returns the corresponding activation function.

        Args:
            act_name: Name of the activation function.
            out_ch: Optional number of output channels. Only used for determining the
                    number of parameters in the PReLU activation function.

        Returns:
            act: PyTorch activation function.

        Experimental results for randomized overdrive neural networks.
        ----------------------
        - ReLU: solid distortion
        - LeakyReLU: somewhat veiled sound
        - Tanh: insane levels of distortion with lots of aliasing (HF)
        - Sigmoid: too gritty to be useful
        - ELU: fading in and out
        - RReLU: really interesting HF noise with a background sound
        - SELU: rolled off soft distortion sound
        - GELU: roomy, not too interesting
        - Softplus: heavily distorted signal but with a very rolled off sound. (nice)
        - Softshrink: super distant sounding and somewhat roomy
        """
        act_name = act_name.lower()
        if act_name == "relu":
            act = nn.ReLU()
        elif act_name == "leakyrelu":
            act = nn.LeakyReLU()
        elif act_name == "tanh":
            act = nn.Tanh()
        elif act_name == "sigmoid":
            act = nn.Sigmoid()
        elif act_name == "elu":
            act = nn.ELU()
        elif act_name == "rrelu":
            act = nn.RReLU()
        elif act_name == "selu":
            act = nn.SELU()
        elif act_name == "gelu":
            act = nn.GELU()
        elif act_name == "softplus":
            act = nn.Softplus()
        elif act_name == "softshrink":
            act = nn.Softshrink()
        elif act_name == "silu" or act_name == "swish":
            act = nn.SiLU()
        elif act_name == "prelu":
            if out_ch is None:
                act = nn.PReLU()
            else:
                act = nn.PReLU(out_ch)
        elif act_name == "prelu1":
            act = nn.PReLU()
        else:
            raise ValueError(f"Invalid activation name: '{act_name}'.")

        return act


class TCN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: List[int],
                 kernel_size: int = 3,
                 strides: Optional[List[int]] = None,
                 padding: Union[str, int, Tuple[int]] = "same",
                 dilations: Optional[List[int]] = None,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 causal: bool = True,
                 cached: bool = False,
                 use_dynamic_bs: bool = True,
                 batch_size: int = 1,
                 use_ln: bool = False,
                 temporal_dims: Optional[List[int]] = None,
                 use_act: bool = True,
                 act_name: str = "prelu",
                 use_res: bool = True,
                 cond_dim: int = 0,
                 use_film_bn: bool = True,  # TODO(cm): check if this should be false
                 debug_mode: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.bias = bias
        self.padding_mode = padding_mode
        self.causal = causal
        self.cached = cached
        self.use_dynamic_bs = use_dynamic_bs
        self.batch_size = batch_size
        self.use_ln = use_ln
        self.temporal_dims = temporal_dims
        self.use_act = use_act
        self.act_name = act_name
        self.use_res = use_res
        self.cond_dim = cond_dim
        self.use_film_bn = use_film_bn
        self.debug_mode = debug_mode

        self.n_blocks = len(out_channels)
        assert self.n_blocks > 0

        if dilations is None:
            dilations = [4 ** idx for idx in range(self.n_blocks)]
            log.info(f"Setting dilations automatically to: {dilations}")
        assert len(dilations) == self.n_blocks
        self.dilations = dilations

        if strides is None:
            strides = [1] * self.n_blocks
            log.info(f"Setting strides   automatically to: {strides}")
        assert len(strides) == self.n_blocks
        self.strides = strides

        if use_ln:
            assert temporal_dims is not None
            assert len(temporal_dims) == self.n_blocks

        self.blocks = nn.ModuleList()
        block_out_ch = None
        for idx, (curr_out_ch, dil, stride) in enumerate(zip(out_channels,
                                                             dilations,
                                                             strides)):
            if idx == 0:
                block_in_ch = in_channels
            else:
                block_in_ch = block_out_ch
            block_out_ch = curr_out_ch

            temp_dim = None
            if temporal_dims is not None:
                temp_dim = temporal_dims[idx]

            self.blocks.append(TCNBlock(block_in_ch,
                                        block_out_ch,
                                        kernel_size,
                                        stride,
                                        padding,
                                        dil,
                                        bias,
                                        padding_mode,
                                        causal,
                                        cached,
                                        use_dynamic_bs,
                                        batch_size,
                                        use_ln,
                                        temp_dim,
                                        use_act,
                                        act_name,
                                        use_res,
                                        cond_dim,
                                        use_film_bn,
                                        debug_mode))

    @tr.jit.export
    def is_conditional(self) -> bool:
        """Returns True if the TCN is conditional, False otherwise."""
        return self.cond_dim > 0

    @tr.jit.export
    def is_cached(self) -> bool:
        """Returns True if the TCN is cached, False otherwise."""
        return self.cached

    @tr.jit.export
    def set_cached(self, cached: bool) -> None:
        """
        Sets the TCN to cached or not cached mode and resets its state.

        Args:
            cached: If True, the TCN is cached. If False, it is not cached.
        """
        self.cached = cached
        for block in self.blocks:
            block.set_cached(cached)

    @tr.jit.export
    def reset(self, batch_size: Optional[int] = None) -> None:
        """
        Resets the TCN's state. If batch_size is provided, the cached padding
        will be resized to match the new batch size.

        Args:
            batch_size: If provided, the cached padding will be resized to match the new
                        batch size.
        """
        for block in self.blocks:
            block.reset(batch_size)

    @tr.jit.export
    def get_delay_samples(self) -> int:
        """
        Returns the number of samples that the TCN delays the output by. This
        should always be 0 when the TCN is causal. This is ill-defined when not
        in cached mode since the output number of samples can be different than the
        input number of samples, so this would typically only be used in cached mode.
        """
        # TODO(cm): verify this
        delay_samples = 0
        for block in self.blocks:
            delay_samples += block.get_delay_samples()
        return delay_samples

    @tr.jit.export
    def calc_receptive_field(self) -> int:
        """Computes the receptive field of the TCN in samples."""
        assert all(_ == 1 for _ in self.strides)  # TODO(cm): add support for dsTCN
        assert self.dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
        rf = self.kernel_size
        for dil in self.dilations[1:]:
            rf += ((self.kernel_size - 1) * dil)
        return rf

    def prepare_for_inference(self) -> None:
        """
        Prepares the TCN for inference by disabling debug mode and ensuring the
        TCN is in cached mode.
        """
        self.debug_mode = False
        for block in self.blocks:
            block.prepare_for_inference()
        self.set_cached(True)
        self.eval()

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        if self.debug_mode:
            assert x.ndim == 3  # (bs, in_ch, samples)
            if self.is_conditional():
                assert cond is not None
                assert cond.shape == (x.size(0), self.cond_dim)  # (bs, cond_dim)
        for block in self.blocks:
            x = block(x, cond)
        return x


if __name__ == '__main__':
    out_channels = [8] * 4
    tcn = TCN(1, out_channels, cond_dim=3, causal=False, cached=False, padding="valid")
    log.info(f"Receptive field: {tcn.calc_receptive_field()}")
    log.info(f"Delay samples:   {tcn.get_delay_samples()}")
    audio = tr.rand((1, 1, 65536))
    cond = tr.rand((1, 3))
    # cond = None
    out = tcn.forward(audio, cond)
    log.info(out.shape)

    script = tr.jit.script(tcn)
