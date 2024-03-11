import logging
import math
import os
import random

import torch as tr
import torch.nn.functional as F
from tqdm import tqdm

from neutone_sdk.sandwich import (
    LinearResampler,
    InplaceLinearResampler,
    Inplace4pHermiteResampler,
)

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def test_linear_resamplers(
    n_trials: int = 1000, in_n_ch: int = 2, out_n_ch: int = 2
) -> None:
    random.seed(42)
    tr.manual_seed(42)
    sampling_rates = [16000, 22050, 32000, 44100, 48000, 88200, 96000]
    buffer_sizes = [64, 128, 256, 512, 1024, 2048]

    resampler = LinearResampler(48000, 48000, 512)
    inplace_resampler = InplaceLinearResampler(in_n_ch, out_n_ch, 48000, 48000, 512)

    for _ in tqdm(range(n_trials)):
        sr_a = random.choice(sampling_rates)
        sr_b = random.choice(sampling_rates)
        in_bs = random.choice(buffer_sizes)

        resampler.set_sample_rates(sr_a, sr_b, in_bs)
        inplace_resampler.set_sample_rates(sr_a, sr_b, in_bs)
        # Check inplace linear resampler internal values are correct for matching the ends exactly
        assert inplace_resampler.x_in[0] == 0.0
        assert inplace_resampler.x_in[-1] == 0.0 or inplace_resampler.x_in[-1] == 1.0

        in_audio = tr.rand((in_n_ch, in_bs))
        in_linear = resampler.process_in(in_audio)
        in_linear_inplace = inplace_resampler.process_in(in_audio)
        out_bs = in_linear.size(1)
        assert in_linear.shape == in_linear_inplace.shape

        # PyTorch interpolation does not match the ends exactly, hence two asserts
        assert tr.allclose(in_linear[:, 1:-1], in_linear_inplace[:, 1:-1], atol=1e-6)
        assert tr.allclose(
            in_linear[:, [0, -1]], in_linear_inplace[:, [0, -1]], atol=1e-3
        )
        in_interpolated = F.interpolate(
            in_audio.unsqueeze(0), out_bs, mode="linear", align_corners=True
        ).squeeze(0)
        # PyTorch interpolation does not match the ends exactly, hence two asserts
        assert tr.allclose(
            in_linear_inplace[:, 1:-1], in_interpolated[:, 1:-1], atol=1e-6
        )
        assert tr.allclose(
            in_linear_inplace[:, [0, -1]], in_interpolated[:, [0, -1]], atol=1e-3
        )
        # Check that the ends match exactly
        assert tr.equal(in_linear_inplace[:, [0, -1]], in_audio[:, [0, -1]])

        out_audio = tr.rand((out_n_ch, out_bs))
        out_linear = resampler.process_out(out_audio)
        out_linear_inplace = inplace_resampler.process_out(out_audio)
        assert out_linear.shape == out_linear_inplace.shape
        assert out_linear.size(1) == in_bs

        # PyTorch interpolation does not match the ends exactly, hence two asserts
        assert tr.allclose(out_linear[:, 1:-1], out_linear_inplace[:, 1:-1], atol=1e-6)
        assert tr.allclose(
            out_linear[:, [0, -1]], out_linear_inplace[:, [0, -1]], atol=1e-3
        )
        out_interpolated = F.interpolate(
            out_audio.unsqueeze(0), in_bs, mode="linear", align_corners=True
        ).squeeze(0)
        # PyTorch interpolation does not match the ends exactly, hence two asserts
        assert tr.allclose(
            out_linear_inplace[:, 1:-1], out_interpolated[:, 1:-1], atol=1e-6
        )
        assert tr.allclose(
            out_linear_inplace[:, [0, -1]], out_interpolated[:, [0, -1]], atol=1e-3
        )
        # Check that the ends match exactly
        assert tr.equal(out_linear_inplace[:, [0, -1]], out_audio[:, [0, -1]])


def _calc_4p_hermite(x: float, y_m1: float, y0: float, y1: float, y2: float) -> float:
    # This is super slow, but the fast version has already been implemented and is being tested using this
    c0 = y0
    c1 = 0.5 * (y1 - y_m1)
    c2 = y_m1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2
    c3 = 0.5 * (y2 - y_m1) + 1.5 * (y0 - y1)
    return ((c3 * x + c2) * x + c1) * x + c0


def test_4p_hermite_resampler(
    n_trials: int = 50, in_n_ch: int = 2, out_n_ch: int = 2
) -> None:
    random.seed(42)
    tr.manual_seed(42)
    sampling_rates = [16000, 22050, 32000, 44100, 48000, 88200, 96000]
    buffer_sizes = [64, 128, 256, 512, 1024, 2048]

    resampler = Inplace4pHermiteResampler(in_n_ch, out_n_ch, 48000, 48000, 512)

    for _ in tqdm(range(n_trials)):
        sr_a = random.choice(sampling_rates)
        sr_b = random.choice(sampling_rates)
        in_bs = random.choice(buffer_sizes)

        resampler.set_sample_rates(sr_a, sr_b, in_bs)
        out_bs = resampler.out_bs

        # Check inplace resampler internal values are correct for matching the first sample
        assert resampler.x_in[0] == 0.0
        assert resampler.x_in[-1] == 0.0 or resampler.x_in[-1] == 1.0

        # Check process_in()
        in_audio = tr.rand((in_n_ch, in_bs))
        in_resampled = resampler.process_in(in_audio)
        assert in_resampled.size(0) == in_n_ch
        # Check that the first sample is equal to the input audio
        assert tr.equal(in_resampled[:, 0], in_audio[:, 0])
        # Check that the last sample is reasonably close to the input audio
        assert tr.allclose(in_resampled[:, -1], in_audio[:, -1], atol=1e-3)

        # Check the 4p cubic hermite spline calculation element-wise
        x = resampler.x_in
        y_m1 = tr.index_select(in_audio, dim=1, index=resampler.y_m1_idx_in)
        y0 = tr.index_select(in_audio, dim=1, index=resampler.y0_idx_in)
        y1 = tr.index_select(in_audio, dim=1, index=resampler.y1_idx_in)
        y2 = tr.index_select(in_audio, dim=1, index=resampler.y2_idx_in)

        for ch_idx in range(in_n_ch):
            for x_idx in range(out_bs):
                y_calc = _calc_4p_hermite(
                    x[x_idx],
                    y_m1[ch_idx, x_idx],
                    y0[ch_idx, x_idx],
                    y1[ch_idx, x_idx],
                    y2[ch_idx, x_idx],
                )
                assert math.isclose(y_calc, in_resampled[ch_idx, x_idx], abs_tol=1e-6)

        # TODO(cm): remove duplication
        # Check process_out()
        out_audio = tr.rand((out_n_ch, out_bs))
        out_resampled = resampler.process_out(out_audio)
        assert out_resampled.size(0) == out_n_ch
        # Check that the first sample is equal to the input audio
        assert tr.equal(out_resampled[:, 0], out_audio[:, 0])
        # Check that the last sample is reasonably close to the input audio
        assert tr.allclose(out_resampled[:, -1], out_audio[:, -1], atol=1e-3)

        # Check the 4p cubic hermite spline calculation element-wise
        x = resampler.x_out
        y_m1 = tr.index_select(out_audio, dim=1, index=resampler.y_m1_idx_out)
        y0 = tr.index_select(out_audio, dim=1, index=resampler.y0_idx_out)
        y1 = tr.index_select(out_audio, dim=1, index=resampler.y1_idx_out)
        y2 = tr.index_select(out_audio, dim=1, index=resampler.y2_idx_out)

        for ch_idx in range(out_n_ch):
            for x_idx in range(in_bs):
                y_calc = _calc_4p_hermite(
                    x[x_idx],
                    y_m1[ch_idx, x_idx],
                    y0[ch_idx, x_idx],
                    y1[ch_idx, x_idx],
                    y2[ch_idx, x_idx],
                )
                assert math.isclose(y_calc, out_resampled[ch_idx, x_idx], abs_tol=1e-6)
