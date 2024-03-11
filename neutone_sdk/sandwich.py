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
        Set the sampling rates of the sandwich. This should be called every time in_sr,
        out_sr, or in_bs changes.
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


class LinearResampler(ResampleSandwich):
    """
    Interpolation-based resampling using the default PyTorch linear interpolation
    implementation.
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


class InplaceLinearResampler(ResampleSandwich):
    """
    Interpolation-based resampling using a custom implementation.
    Does not dynamically allocate memory and is ~40% faster than the PyTorch
    implementation for common sampling rates.
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
        """
        Sets the sampling rates of the resampler. This should be called every time
        in_sr, out_sr, or in_bs changes. Allocates all memory used by the resampler.
        """
        # Repeated code due to TorchScript limitations
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.in_bs = in_bs
        self.out_bs = math.ceil(self.in_bs * self.out_sr / self.in_sr)
        if self.use_debug_mode:
            assert self.out_bs >= 2

        self.x_in, self.y0_idx_in, self.y1_idx_in = self.calc_x_and_indices(
            self.in_bs, self.out_bs
        )
        self.y0_in = tr.zeros((self.in_n_ch, self.out_bs))
        self.y1_in = tr.zeros((self.in_n_ch, self.out_bs))
        self.x_out, self.y0_idx_out, self.y1_idx_out = self.calc_x_and_indices(
            self.out_bs, self.in_bs
        )
        self.y0_out = tr.zeros((self.out_n_ch, self.in_bs))
        self.y1_out = tr.zeros((self.out_n_ch, self.in_bs))

    def _process_2p_linear(
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
        # Calc interpolated y using y0 as storage
        tr.sub(y1, y0, out=y1)
        tr.mul(y1, x, out=y1)
        tr.add(y0, y1, out=y0)
        return y0

    def process_in(self, y: Tensor) -> Tensor:
        return self._process_2p_linear(
            y,
            self.in_n_ch,
            self.in_bs,
            self.x_in,
            self.y0_idx_in,
            self.y1_idx_in,
            self.y0_in,
            self.y1_in,
        )

    def process_out(self, y: Tensor) -> Tensor:
        return self._process_2p_linear(
            y,
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
        """
        Calculates x values and y0 and y1 index locations for interpolating between the
        given input and output buffer sizes.
        """
        # Prevents floating point errors
        scaling_factor = (in_bs - 1) / (out_bs - 1) + 1e-12
        x = tr.arange(0, out_bs) * scaling_factor
        y0_idx = tr.floor(x).to(tr.long)
        y1_idx = y0_idx + 1
        y1_idx = tr.clip(y1_idx, 0, in_bs - 1)  # Prevent overflow
        x = tr.clip(x - y0_idx, 0.0, 1.0)  # Prevents floating point errors
        # This ensures corners match exactly
        x[0] = 0.0
        x[-1] = tr.round(x[-1])
        return x, y0_idx, y1_idx


class Inplace4pHermiteResampler(InplaceLinearResampler):
    """
    4-point cubic hermite spline interpolation.
    Implementation taken from "Polynomial Interpolators for High-Quality Resampling of
    Oversampled Audio" by Olli Niemitalo
    (http://yehar.com/blog/wp-content/uploads/2009/08/deip.pdf).
    Does not dynamically allocate memory and is only ~1.8x slower than the
    InplaceLinearResampler. For comparison, sinc interpolation is ~4.5x slower and
    requires dynamic memory allocations.
    The 2nd, -2nd, and sometimes -1st samples will be slightly off since this
    interpolator requires 4 points, but we assume a repeated sample when these are not
    available at the ends rather than adding 2-samples of delay.
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
        # Buffers required for process_in
        self.y_m1_idx_in = None
        self.y2_idx_in = None
        self.y_m1_in = None
        self.y2_in = None
        self.c1_in = None
        self.c2_in = None
        self.c3_in = None
        # Buffers required for process_out
        self.y_m1_idx_out = None
        self.y2_idx_out = None
        self.y_m1_out = None
        self.y2_out = None
        self.c1_out = None
        self.c2_out = None
        self.c3_out = None
        # Constants
        self.const_0p5 = tr.tensor(0.5)
        self.const_1p5 = tr.tensor(1.5)
        self.const_2p0 = tr.tensor(2.0)
        self.const_2p5 = tr.tensor(2.5)
        self.const_3p0 = tr.tensor(3.0)
        super().__init__(in_n_ch, out_n_ch, in_sr, out_sr, in_bs, use_debug_mode)

    def set_sample_rates(self, in_sr: int, out_sr: int, in_bs: int) -> None:
        """
        Sets the sampling rates of the resampler. This should be called every time
        in_sr, out_sr, or in_bs changes. Allocates all memory used by the resampler.
        """
        # Repeated code due to TorchScript limitations
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.in_bs = in_bs
        self.out_bs = math.ceil(self.in_bs * self.out_sr / self.in_sr)
        if self.use_debug_mode:
            assert self.out_bs > 4

        self.x_in, self.y0_idx_in, self.y1_idx_in = self.calc_x_and_indices(
            self.in_bs, self.out_bs
        )
        self.y0_in = tr.zeros((self.in_n_ch, self.out_bs))
        self.y1_in = tr.zeros((self.in_n_ch, self.out_bs))
        self.x_out, self.y0_idx_out, self.y1_idx_out = self.calc_x_and_indices(
            self.out_bs, self.in_bs
        )
        self.y0_out = tr.zeros((self.out_n_ch, self.in_bs))
        self.y1_out = tr.zeros((self.out_n_ch, self.in_bs))

        self.y_m1_idx_in = tr.clip(self.y0_idx_in - 1, 0, self.in_bs - 1)
        self.y2_idx_in = tr.clip(self.y1_idx_in + 1, 0, self.in_bs - 1)
        self.y_m1_in = tr.zeros((self.in_n_ch, self.out_bs))
        self.y2_in = tr.zeros((self.in_n_ch, self.out_bs))
        self.c1_in = tr.zeros((self.in_n_ch, self.out_bs))
        self.c2_in = tr.zeros((self.in_n_ch, self.out_bs))
        self.c3_in = tr.zeros((self.in_n_ch, self.out_bs))

        self.y_m1_idx_out = tr.clip(self.y0_idx_out - 1, 0, self.out_bs - 1)
        self.y2_idx_out = tr.clip(self.y1_idx_out + 1, 0, self.out_bs - 1)
        self.y_m1_out = tr.zeros((self.out_n_ch, self.in_bs))
        self.y2_out = tr.zeros((self.out_n_ch, self.in_bs))
        self.c1_out = tr.zeros((self.out_n_ch, self.in_bs))
        self.c2_out = tr.zeros((self.out_n_ch, self.in_bs))
        self.c3_out = tr.zeros((self.out_n_ch, self.in_bs))

    def _process_4p_hermite(
        self,
        y: Tensor,
        n_ch: int,
        in_bs: int,
        x: Tensor,
        y_m1_idx: Tensor,
        y0_idx: Tensor,
        y1_idx: Tensor,
        y2_idx: Tensor,
        y_m1: Tensor,
        y0: Tensor,
        y1: Tensor,
        y2: Tensor,
        c1: Tensor,
        c2: Tensor,
        c3: Tensor,
    ) -> Tensor:
        if self.use_debug_mode:
            assert y.shape == (n_ch, in_bs)
        if not self.is_resampling():
            return y
        tr.index_select(y, dim=1, index=y_m1_idx, out=y_m1)
        tr.index_select(y, dim=1, index=y0_idx, out=y0)
        tr.index_select(y, dim=1, index=y1_idx, out=y1)
        tr.index_select(y, dim=1, index=y2_idx, out=y2)
        # Calc c2 using c1 and c3 as temporary storage
        # y[-1] - 5/2.0*y[0] + 2*y[1] - 1/2.0*y[2]
        tr.mul(self.const_2p5, y0, out=c2)
        tr.sub(y_m1, c2, out=c2)
        tr.mul(self.const_2p0, y1, out=c1)
        tr.add(c2, c1, out=c2)
        tr.mul(self.const_0p5, y2, out=c3)
        tr.sub(c2, c3, out=c2)
        # Calc c3 using c1 as temporary storage
        # 1/2.0*(y[2]-y[-1]) + 3/2.0*(y[0]-y[1])
        tr.sub(y2, y_m1, out=c3)
        tr.mul(self.const_0p5, c3, out=c3)
        tr.sub(y0, y1, out=c1)
        tr.mul(self.const_1p5, c1, out=c1)
        tr.add(c3, c1, out=c3)
        # Calc c1
        # 1/2.0*(y[1]-y[-1]);
        tr.sub(y1, y_m1, out=c1)
        tr.mul(self.const_0p5, c1, out=c1)
        c0 = y0
        # Calc interpolated y using y1 as storage
        # ((c3*x+c2)*x+c1)*x+c0
        tr.mul(c3, x, out=y1)
        tr.add(y1, c2, out=y1)
        tr.mul(y1, x, out=y1)
        tr.add(y1, c1, out=y1)
        tr.mul(y1, x, out=y1)
        tr.add(y1, c0, out=y1)
        return y1

    def _process_4p_hermite_opt(
        self,
        y: Tensor,
        n_ch: int,
        in_bs: int,
        x: Tensor,
        y_m1_idx: Tensor,
        y0_idx: Tensor,
        y1_idx: Tensor,
        y2_idx: Tensor,
        y_m1: Tensor,
        y0: Tensor,
        y1: Tensor,
        y2: Tensor,
        c1: Tensor,
        c2: Tensor,
        c3: Tensor,
    ) -> Tensor:
        """
        Optimized version of _process_4p_hermite that uses 9 add/sub (instead of 10) and
        6 multiplications (instead of 9).
        Found on stack overflow: https://stackoverflow.com/a/72122178
        """
        if self.use_debug_mode:
            assert y.shape == (n_ch, in_bs)
        if not self.is_resampling():
            return y
        tr.index_select(y, dim=1, index=y_m1_idx, out=y_m1)
        tr.index_select(y, dim=1, index=y0_idx, out=y0)
        tr.index_select(y, dim=1, index=y1_idx, out=y1)
        tr.index_select(y, dim=1, index=y2_idx, out=y2)
        # Calc diff using c2 as storage
        # diff = y[0] - y[1]
        tr.sub(y0, y1, out=c2)
        # Calc c1
        # c1 = y[1] - y[-1]
        tr.sub(y1, y_m1, out=c1)
        # Calc c3
        # c3 = y[2] - y[-1] + 3 * diff;
        tr.mul(self.const_3p0, c2, out=c3)
        tr.add(c3, y2, out=c3)
        tr.sub(c3, y_m1, out=c3)
        # Calc c2
        # c2 = -(2 * diff + c1 + c3)
        tr.mul(self.const_2p0, c2, out=c2)
        tr.add(c2, c1, out=c2)
        tr.add(c2, c3, out=c2)
        tr.neg(c2, out=c2)
        # Calc interpolated y using y1 as storage
        # 0.5 * ((c3 * x + c2) * x + c1) * x + y[0]
        tr.mul(c3, x, out=y1)
        tr.add(y1, c2, out=y1)
        tr.mul(y1, x, out=y1)
        tr.add(y1, c1, out=y1)
        tr.mul(self.const_0p5, y1, out=y1)
        tr.mul(y1, x, out=y1)
        tr.add(y1, y0, out=y1)
        return y1

    def process_in(self, y: Tensor) -> Tensor:
        return self._process_4p_hermite_opt(
            y,
            self.in_n_ch,
            self.in_bs,
            self.x_in,
            self.y_m1_idx_in,
            self.y0_idx_in,
            self.y1_idx_in,
            self.y2_idx_in,
            self.y_m1_in,
            self.y0_in,
            self.y1_in,
            self.y2_in,
            self.c1_in,
            self.c2_in,
            self.c3_in,
        )

    def process_out(self, y: Tensor) -> Tensor:
        return self._process_4p_hermite_opt(
            y,
            self.out_n_ch,
            self.out_bs,
            self.x_out,
            self.y_m1_idx_out,
            self.y0_idx_out,
            self.y1_idx_out,
            self.y2_idx_out,
            self.y_m1_out,
            self.y0_out,
            self.y1_out,
            self.y2_out,
            self.c1_out,
            self.c2_out,
            self.c3_out,
        )
