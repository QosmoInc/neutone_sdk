import logging
import math
import os
from abc import abstractmethod, ABC
from typing import Tuple

import torch as tr
import torch.nn.functional as F
from torch import Tensor, nn
from torchaudio.transforms import Resample

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ChannelNormalizerSandwich(nn.Module):
    """
    Converts between mono and stereo channels as required without allocating memory.
    """

    def __init__(self, use_debug_mode: bool = True) -> None:
        super().__init__()
        self.use_debug_mode = use_debug_mode
        self.half_scalar = tr.tensor(0.5)

    def forward(self, x: Tensor, should_be_mono: bool, out_buffer: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert x.ndim == 2
            assert x.size(0) <= 2
            assert out_buffer.ndim == 2
            assert out_buffer.size(0) == 2
            assert out_buffer.size(1) >= x.size(1)
        n_ch = x.size(0)
        n_samples = x.size(1)
        if should_be_mono and n_ch == 2:
            out = out_buffer[0:1, 0:n_samples]
            tr.add(x[0:1, :], x[1:2, :], out=out)
            tr.mul(out, self.half_scalar, out=out)
            x = out
        elif not should_be_mono and n_ch == 1:
            out_buffer[0:1, 0:n_samples] = x
            out_buffer[1:2, 0:n_samples] = x
            x = out_buffer
        return x


class ResampleSandwich(ABC, nn.Module):
    def __init__(
        self, in_sr: int, out_sr: int, in_bs: int, use_debug_mode: bool = True
    ) -> None:
        """
        Common interface for resampling sandwiches.

        Args:
            in_sr: incoming sampling rate
            out_sr: desired sampling rate
            in_bs: incoming buffer size
            use_debug_mode: enable debug mode
        """
        super().__init__()
        self.use_debug_mode = use_debug_mode

        if self.use_debug_mode:
            assert in_sr > 0
            assert out_sr > 0
            assert in_bs >= 2

        self.in_sr = in_sr
        self.out_sr = out_sr
        self.in_bs = in_bs

        self.out_bs = None
        self.set_sample_rates(in_sr, out_sr, in_bs)

    def set_sample_rates(self, in_sr: int, out_sr: int, in_bs: int) -> None:
        """
        Set the sampling rates of the sandwich. This should be called every time in_sr, out_sr, or in_bs changes.
        """
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.in_bs = in_bs
        self.out_bs = math.ceil(self.in_bs * self.out_sr / self.in_sr)
        if self.use_debug_mode:
            assert self.out_bs >= 2

    def is_resampling(self) -> bool:
        return self.in_bs != self.out_bs

    @abstractmethod
    def process_in(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def process_out(self, x: Tensor) -> Tensor:
        pass


# TODO(cm): make this torchscript compatible
class PTResampler(ResampleSandwich):
    """
    Antialiasing resampling using the default PyTorch audio resampling implementation.
    Slower, dynamically allocates memory, and is not TorchScript compatible.
    """

    def __init__(
        self,
        in_sr: int,
        out_sr: int,
        in_bs: int,
        resampling_method: str = "sinc_interpolation",
        align_corners: bool = True,
        use_debug_mode: bool = True,
    ) -> None:
        self.resampling_method = resampling_method
        self.align_corners = align_corners
        self.in_resampler = None
        self.out_resampler = None
        super().__init__(in_sr, out_sr, in_bs, use_debug_mode)

    def set_sample_rates(self, in_sr: int, out_sr: int, in_bs: int) -> None:
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.in_bs = in_bs

        self.in_resampler = Resample(
            orig_freq=self.in_sr,
            new_freq=self.out_sr,
            resampling_method=self.resampling_method,
        )
        self.out_resampler = Resample(
            orig_freq=self.out_sr,
            new_freq=self.in_sr,
            resampling_method=self.resampling_method,
        )

        tmp = self.in_resampler(tr.zeros((2, self.in_bs)))
        self.out_bs = tmp.size(1)
        if self.use_debug_mode:
            assert self.out_bs >= 2

    def process_in(self, x: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert x.ndim == 2
            assert x.size(1) == self.in_bs
        if self.is_resampling():
            corner_1_value = x[:, 0]
            corner_2_value = x[:, -1]
            x = self.in_resampler(x)
            if self.align_corners:
                x[:, 0] = corner_1_value
                x[:, -1] = corner_2_value
        return x

    def process_out(self, x: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert x.ndim == 2
            assert x.size(1) == self.out_bs
        if self.is_resampling():
            corner_1_value = x[:, 0]
            corner_2_value = x[:, -1]
            x = self.out_resampler(x)
            if x.size(1) > self.in_bs:
                x = x[:, : self.in_bs]
            if self.align_corners:
                x[:, 0] = corner_1_value
                x[:, -1] = corner_2_value
        return x


class InterpolationResampler(ResampleSandwich):
    """
    Interpolation-based resampling using the default PyTorch linear interpolation implementation.
    Dynamically allocates memory.
    """

    def _process(self, x: Tensor, in_bs: int, out_bs: int) -> Tensor:
        if self.use_debug_mode:
            assert x.ndim == 2
            assert x.size(1) == in_bs
        if self.is_resampling():
            x = x.unsqueeze(0)
            x = F.interpolate(x, out_bs, mode="linear", align_corners=True)
            x = x.squeeze(0)
        return x

    def process_in(self, x: Tensor) -> Tensor:
        return self._process(x, self.in_bs, self.out_bs)

    def process_out(self, x: Tensor) -> Tensor:
        return self._process(x, self.out_bs, self.in_bs)


class InplaceInterpolationResampler(ResampleSandwich):
    """
    Interpolation-based resampling using a custom implementation.
    Does not dynamically allocate memory and is ~40% faster than the PyTorch implementation for common sampling rates.
    """

    def __init__(
        self,
        in_n_ch: int,
        out_n_ch: int,
        in_sr: int,
        out_sr: int,
        in_bs: int,
        use_debug_mode: bool = True,
    ) -> None:
        self.in_n_ch = in_n_ch
        self.out_n_ch = out_n_ch

        # Buffers required for process_in
        self.x_in = None
        self.y0_idx_in = None
        self.y1_idx_in = None
        self.y0_in = None
        self.y1_in = None
        # Buffers required for process_out
        self.x_out = None
        self.y0_idx_out = None
        self.y1_idx_out = None
        self.y0_out = None
        self.y1_out = None

        super().__init__(in_sr, out_sr, in_bs, use_debug_mode)

    def set_sample_rates(self, in_sr: int, out_sr: int, in_bs: int) -> None:
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.in_bs = in_bs
        self.out_bs = math.ceil(self.in_bs * self.out_sr / self.in_sr)
        if self.use_debug_mode:
            assert self.out_bs >= 2

        self.x_in, self.y0_idx_in, self.y1_idx_in = self.calc_x_and_indices(self.in_bs, self.out_bs)
        self.y0_in = tr.zeros((self.in_n_ch, self.out_bs))
        self.y1_in = tr.zeros((self.in_n_ch, self.out_bs))
        self.x_out, self.y0_idx_out, self.y1_idx_out = self.calc_x_and_indices(self.out_bs, self.in_bs)
        self.y0_out = tr.zeros((self.out_n_ch, self.in_bs))
        self.y1_out = tr.zeros((self.out_n_ch, self.in_bs))

    def _process(
        self,
        y: Tensor,
        n_ch: int,
        in_bs: int,
        x: Tensor,
        y0_idx: Tensor,
        y1_idx: Tensor,
        y0: Tensor,
        y1: Tensor,
    ) -> Tensor:
        if self.use_debug_mode:
            assert y.shape == (n_ch, in_bs)
        if not self.is_resampling():
            return y
        tr.index_select(y, dim=1, index=y0_idx, out=y0)
        tr.index_select(y, dim=1, index=y1_idx, out=y1)
        tr.sub(y1, y0, out=y1)
        tr.mul(y1, x, out=y1)
        tr.add(y0, y1, out=y0)
        return y0

    # def _process_hermite(
    #     self,
    #     y: Tensor,
    #     n_ch: int,
    #     in_bs: int,
    #     x: Tensor,
    #     y0_idx: Tensor,
    #     y1_idx: Tensor,
    #     y2_idx: Tensor,
    #     y3_idx: Tensor,
    #     y0: Tensor,
    #     y1: Tensor,
    #     y2: Tensor,
    #     y3: Tensor,
    # ) -> Tensor:
    #     if self.use_debug_mode:
    #         assert y.shape == (n_ch, in_bs)
    #     if not self.is_resampling():
    #         return y
    #     tr.index_select(y, dim=1, index=y0_idx, out=y0)
    #     tr.index_select(y, dim=1, index=y1_idx, out=y1)
    #     tr.index_select(y, dim=1, index=y2_idx, out=y2)
    #     tr.index_select(y, dim=1, index=y3_idx, out=y3)
    #     c0 = y0
    #     c1 = 0.5 * tr.sub(y1, y0, out=y1)
    #     tr.sub(y1, y0, out=y1)
    #     tr.mul(y1, x, out=y1)
    #     tr.add(y0, y1, out=y0)
    #     return y0

    def process_in(self, x: Tensor) -> Tensor:
        return self._process(
            x,
            self.in_n_ch,
            self.in_bs,
            self.x_in,
            self.y0_idx_in,
            self.y1_idx_in,
            self.y0_in,
            self.y1_in,
        )

    def process_out(self, x: Tensor) -> Tensor:
        return self._process(
            x,
            self.out_n_ch,
            self.out_bs,
            self.x_out,
            self.y0_idx_out,
            self.y1_idx_out,
            self.y0_out,
            self.y1_out,
        )

    @staticmethod
    def calc_x_and_indices(in_bs: int, out_bs: int) -> Tuple[Tensor, Tensor, Tensor]:
        scaling_factor = (in_bs - 1) / (out_bs - 1)
        x = tr.arange(0, out_bs) * scaling_factor
        y0_idx = tr.floor(x).to(tr.long)
        y0_idx = tr.clip(y0_idx, 0, in_bs - 1)  # Prevents floating point errors
        y1_idx = tr.ceil(x).to(tr.long)
        y1_idx = tr.clip(y1_idx, 0, in_bs - 1)  # Prevents floating point errors
        x = tr.clip(x - y0_idx, 0.0, 1.0)  # Prevents floating point errors
        return x, y0_idx, y1_idx
