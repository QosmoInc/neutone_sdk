import logging
import os
from typing import Optional, Tuple, Union

import torch as tr
import torch.nn.functional as F
from torch import Tensor
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class PaddingCached(nn.Module):
    def __init__(self,
                 n_ch: int,
                 padding_l: int,
                 padding_r: int = 0,
                 use_dynamic_bs: bool = True,
                 batch_size: int = 1,
                 debug_mode: bool = True) -> None:
        """
        Cached padding for cached convolutions. Handles dynamic batch sizes by default
        at the expense of dynamic memory allocations. TorchScript compatible.

        Args:
            n_ch: Number of channels.
            padding_l: Number of padding samples on the left side.
            padding_r: Number of padding samples on the right side.
            use_dynamic_bs: If True, the padding will dynamically change batch size to
                            match the input.
            batch_size: If known, the initial batch size can be specified here to avoid
                        dynamic changes.
            debug_mode: If True, assert statements are enabled.
        """
        super().__init__()
        if debug_mode:
            assert n_ch > 0
            assert batch_size > 0
        self.n_ch = n_ch
        self.padding_l = padding_l
        self.padding_r = padding_r
        self.use_dynamic_bs = use_dynamic_bs
        self.debug_mode = debug_mode
        self.register_buffer("pad_l_buf", tr.zeros((batch_size, n_ch, padding_l)))
        self.register_buffer("pad_r_buf", tr.zeros((batch_size, n_ch, padding_r)))

    @tr.jit.export
    def reset(self, batch_size: Optional[int] = None) -> None:
        """
        Resets the padding's state. If batch_size is provided, the cached padding will
        be resized to match the new batch size.

        Args:
            batch_size: If provided, the cached padding will be resized to match the new
                        batch size.
        """
        if batch_size is not None:
            self.pad_l_buf = self.pad_l_buf.new_zeros(
                (batch_size, self.n_ch, self.padding_l))
            self.pad_r_buf = self.pad_r_buf.new_zeros(
                (batch_size, self.n_ch, self.padding_r))
        else:
            self.pad_l_buf.zero_()
            self.pad_r_buf.zero_()

    def prepare_for_inference(self) -> None:
        """
        Prepares the padding for inference by disabling debug mode. This method is not
        exported to TorchScript and should be called before converting a model to
        TorchScript since this implies it is going to be used for inference.
        """
        self.debug_mode = False
        self.reset()
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the padding to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, in_ch, in_samples).

        Returns:
            Padded input tensor of shape (batch_size, in_ch, in_samples + padding_l
            + padding_r).
        """
        if self.debug_mode:
            assert x.ndim == 3  # (batch_size, in_ch, samples)
        # We support padding == 0 for convolutions with kernel size of 1
        if self.padding_l == 0 and self.padding_r == 0:
            return x

        # Dynamic batch size resizing
        bs = x.size(0)
        if self.use_dynamic_bs and bs != self.pad_l_buf.size(0):
            self.reset(bs)
        elif self.debug_mode:
            assert bs == self.pad_l_buf.size(0)

        # Left padding
        if self.padding_l > 0:
            x = tr.cat([self.pad_l_buf, x], dim=-1)
            self.pad_l_buf = x[..., -self.padding_l:]
        # Right padding
        if self.padding_r > 0:
            x = tr.cat([x, self.pad_r_buf], dim=-1)
            # We need to account for left padding since it may have been added first
            self.pad_r_buf = x[..., self.padding_l:self.padding_l + self.padding_r]
        return x


class Conv1dGeneral(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Union[str, int, Tuple[int]] = "same",
                 dilation: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 causal: bool = True,
                 cached: bool = False,
                 use_dynamic_bs: bool = True,
                 batch_size: int = 1,
                 debug_mode: bool = True) -> None:
        """
        Generalized 1D convolution that supports causal convolutions and cached
        convolutions. The convolution can be toggled to be cached or not at any point in
        time via the method `set_cached()`. Behaves identically to torch.nn.Conv1d when
        not in cached mode and causal is False. When causal is True, the convolution is
        padded on the left side only. When cached and not causal, the convolution delays
        the output by `get_delay_samples()` samples. TorchScript compatible.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to both sides of the input. Can be 'same', 'valid',
                     or an integer.
            padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'.
            dilation: Spacing between kernel elements.
            bias: If True, adds a learnable bias to the output.
            causal: If True, the convolution is causal.
            cached: If True, the convolution is cached.
            use_dynamic_bs: If True, the convolution will support any batch size while
                            in cached mode at the expense of dynamic memory allocations.
            batch_size: If known, the initial batch size can be specified here to avoid
                        dynamic changes.
            debug_mode: If True, assert statements are enabled.
        """
        super().__init__()
        if stride != 1:
            raise NotImplementedError("Stride > 1 has not been implemented yet")
        self.in_channels = in_channels
        self.padding_mode = padding_mode
        self.causal = causal
        self.cached = cached
        self.debug_mode = debug_mode

        padded_kernel_size = (kernel_size - 1) * dilation
        padding_l, padding_r = self._calc_padding(
            kernel_size, stride, padding, dilation, causal)
        # The left padding required for cached mode is the maximum of the kernel size
        # and the specified left padding for the convolution.
        padding_l_cached = max(padded_kernel_size, padding_l)

        self.padded_kernel_size = padded_kernel_size
        self.padding_l = padding_l
        self.padding_r = padding_r

        # The trainable weights are contained inside a torch.nn.Conv1d module
        self.conv1d = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size=(kernel_size,),
                                stride=(stride,),
                                padding=0,
                                dilation=(dilation,),
                                bias=bias,
                                padding_mode=padding_mode)
        if padding_mode == "zeros":
            # After setting the torch conv padding mode, we need to change it to be
            # compatible with F.pad(). It's strange that this is inconsistent in torch.
            self.padding_mode = "constant"
        self.padding_cached = PaddingCached(in_channels,
                                            padding_l_cached,
                                            padding_r,
                                            use_dynamic_bs=use_dynamic_bs,
                                            batch_size=batch_size,
                                            debug_mode=debug_mode)

    def _calc_padding(self,
                      kernel_size: int,
                      stride: int,
                      padding: Union[str, int, Tuple[int]],
                      dilation: int,
                      causal: bool) -> Tuple[int, int]:
        """
        Calculates the left and right padding for the convolution.

        Args:
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to both sides of the input. Can be 'same', 'valid',
                     or an integer.
            dilation: Spacing between kernel elements.
            causal: If True, the convolution is causal.

        Returns:
            Tuple of left and right padding.
        """
        # Unpack tuple padding if necessary
        if isinstance(padding, tuple) and len(padding) == 1:
            padding = padding[0]
        padded_kernel_size = (kernel_size - 1) * dilation
        if padding == "valid":
            padding_l = 0
            padding_r = 0
        elif padding == "same":
            assert stride == 1, "If padding is 'same', stride must be 1"
            if causal:
                padding_l = padded_kernel_size
                padding_r = 0
            elif padded_kernel_size % 2 == 0:
                padding_l = padded_kernel_size // 2
                padding_r = padded_kernel_size // 2
            else:
                padding_l = padded_kernel_size // 2
                # Favor right padding over left padding to match torch.nn.Conv1d
                padding_r = padded_kernel_size // 2 + 1
        else:
            assert isinstance(padding, int)
            assert padding >= 0
            if causal:
                padding_l = padding
                padding_r = 0
            else:
                padding_l = padding
                padding_r = padding

        return padding_l, padding_r

    @tr.jit.export
    def is_cached(self) -> bool:
        """Returns True if the convolution is cached, False otherwise."""
        return self.cached

    @tr.jit.export
    def set_cached(self, cached: bool) -> None:
        """
        Sets the convolution to cached or not cached mode and resets its state.

        Args:
            cached: If True, the convolution is cached. If False, it is not cached.
        """
        self.cached = cached
        # Batch size needs to be provided for TorchScript
        self.reset(batch_size=None)

    @tr.jit.export
    def reset(self, batch_size: Optional[int] = None) -> None:
        """
        Resets the convolution's state. If batch_size is provided, the cached padding
        will be resized to match the new batch size.

        Args:
            batch_size: If provided, the cached padding will be resized to match the new
                        batch size.
        """
        self.padding_cached.reset(batch_size)

    @tr.jit.export
    def get_delay_samples(self) -> int:
        """
        Returns the number of samples that the convolution delays the output by. This
        should always be 0 when the convolution is causal. This is ill-defined when not
        in cached mode since the output number of samples can be different than the
        input number of samples, so this would typically only be used in cached mode.
        """
        if not self.is_cached():
            log.warning("`get_delay_samples()` is ill-defined when not in cached mode "
                        "since the output number of samples can be different than the "
                        "input number of samples.")
        return self.padding_r

    def prepare_for_inference(self) -> None:
        """
        Prepares the convolution for inference by disabling debug mode and ensuring the
        convolution is in cached mode. This method is not exported to TorchScript and
        should be called before converting a model to TorchScript since this implies
        it is going to be used for inference.
        """
        if not self.is_cached():
            log.info(f"Converting Conv1dGeneral to cached in prepare_for_inference()")
            self.set_cached(True)
        self.debug_mode = False
        self.padding_cached.prepare_for_inference()
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the convolution to the input tensor. If the convolution is cached, the
        output tensor will always be the same size as the input tensor.

        Args:
            x: Input tensor of shape (batch_size, in_ch, in_samples).

        Returns:
            Output tensor of shape (batch_size, out_ch, out_samples).
        """
        if self.debug_mode:
            assert x.ndim == 3  # (batch_size, in_ch, samples)
            assert x.size(1) == self.in_channels
        n_samples = x.size(-1)
        if self.is_cached():
            x = self.padding_cached(x)
        elif self.padding_l > 0 or self.padding_r > 0:
            # TODO(cm): prevent dynamic memory allocations here
            x = F.pad(x, (self.padding_l, self.padding_r), mode=self.padding_mode)
        x = self.conv1d(x)
        if self.is_cached():
            if self.causal:
                x = self.causal_crop(x, n_samples)
            elif self.padding_r > 0:
                # If cached, but non-causal, we need to remove the right padding which
                # is the non-causal portion of the convolution's output.
                x = self.right_offset_crop(x, n_samples, self.padding_r)
        return x

    @staticmethod
    def center_crop(x: Tensor, length: int) -> Tensor:
        """
        Crops the input tensor to the specified length by removing samples from the
        beginning and end of the input tensor.

        Args:
            x: Input tensor of shape (..., n_samples).
            length: Length of the output tensor. Must be less than n_samples.

        Returns:
            Output tensor of shape (..., length).
        """
        if x.size(-1) != length:
            assert x.size(-1) > length
            start = (x.size(-1) - length) // 2
            if x.size(-1) % 2 != 0:
                # torch.nn.Conv1d favors right padding over left padding in this case
                start += 1
            stop = start + length
            x = x[..., start:stop]
        return x

    @staticmethod
    def causal_crop(x: Tensor, length: int) -> Tensor:
        """
        Crops the input tensor to the specified length by removing samples from the
        beginning of the input tensor.

        Args:
            x: Input tensor of shape (..., n_samples).
            length: Length of the output tensor. Must be less than n_samples.

        Returns:
            Output tensor of shape (..., length).
        """
        if x.size(-1) != length:
            assert x.size(-1) > length
            x = x[..., -length:]
        return x

    @staticmethod
    def right_offset_crop(x: Tensor, length: int, right_offset: int) -> Tensor:
        """
        Crops the input tensor to the specified length by removing exactly
        `right_offset` number of samples from the right of the input.

        Args:
            x: Input tensor of shape (..., n_samples).
            length: Length of the output tensor. Must be less than n_samples.
            right_offset: Number of samples to remove from the right of the input.

        Returns:
            Output tensor of shape (..., length).
        """
        if x.size(-1) != length:
            assert x.size(-1) >= length + right_offset
            stop = x.size(-1) - right_offset
            start = stop - length
            x = x[..., start:stop]
        return x
