import logging
import math
import os
from typing import Optional, List

import torch as tr
from torch import Tensor, nn

from neutone_sdk import WaveformToWaveformMetadata, validate_waveform
from neutone_sdk.constants import DEFAULT_DAW_SR, DEFAULT_DAW_BS
from neutone_sdk.queues import CircularInplaceTensorQueue
from neutone_sdk.sandwich import (
    ChannelNormalizerSandwich,
    InplaceLinearResampler,
    Inplace4pHermiteResampler,
)

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SampleQueueWrapper(nn.Module):
    def __init__(
        self,
        w2w_base: "WaveformToWaveformBase",
        daw_sr: int = DEFAULT_DAW_SR,
        model_sr: Optional[int] = None,
        daw_bs: int = DEFAULT_DAW_BS,
        model_bs: Optional[int] = None,
        use_debug_mode: bool = True,
    ) -> None:
        """
        Creates a SampleQueueWrapper (SQW) which wraps a WaveformToWaveformBase Neutone model to make it compatible
        with varying DAW sampling rates, buffer sizes, and also delay reporting, and multithreading.

        Related issue: https://github.com/QosmoInc/neutone_sdk/issues/6
        """
        super().__init__()
        self.use_debug_mode = use_debug_mode

        self.w2w_base = w2w_base
        self.in_n_ch = 1 if self.is_input_mono() else 2
        self.out_n_ch = 1 if self.is_output_mono() else 2

        self.channel_normalizer = ChannelNormalizerSandwich(
            use_debug_mode=use_debug_mode
        )
        self.resample_sandwich = Inplace4pHermiteResampler(
            self.in_n_ch,
            self.out_n_ch,
            daw_sr,
            daw_sr,  # Tmp sample rate values
            daw_bs,
            use_debug_mode=use_debug_mode,
        )
        self.params_resample_sandwich = InplaceLinearResampler(
            self.w2w_base.MAX_N_PARAMS,
            self.w2w_base.MAX_N_PARAMS,
            daw_sr,
            daw_sr,  # Tmp sample rate value
            daw_bs,
            use_debug_mode=use_debug_mode,
        )

        self.daw_sr = daw_sr
        self.model_sr = model_sr
        self.daw_bs = daw_bs
        self.io_bs = daw_bs  # Temp value for typing
        self.model_bs = model_bs
        self.seen_n = 0
        self.is_queue_saturated = False
        self.saturation_n = None

        self.in_queue = None
        self.params_queue = None
        self.out_queue = None

        self.daw_buffer = None
        self.model_in_buffer = None
        self.params_buffer = None
        self.io_out_buffer = None
        self.bt_out_buffer = None

        self.set_daw_sample_rate_and_buffer_size(daw_sr, daw_bs, model_sr, model_bs)

    @staticmethod
    def select_best_model_sr(daw_sr: int, native_sample_rates: List[int]) -> int:
        """
        Given a DAW sampling rate and a list of all the sampling rates a Neutone model supports (usually only one, or
        an empty list indicates all sampling rates are supported), determine the optimal sampling rate to use.
        """
        # Avoid resampling whenever possible
        if not native_sample_rates:
            return daw_sr
        if daw_sr in native_sample_rates:
            return daw_sr
        # Resampling is unavoidable
        if len(native_sample_rates) == 1:
            return native_sample_rates[0]
        # TODO(cm): combine this with selecting the buffer size to be smarter
        # TODO(cm): prefer upsampling if the buffer sizes allow it
        # This is a workaround for torchscript not supporting lambda functions
        diffs = [abs(sr - daw_sr) for sr in native_sample_rates]
        min_idx = diffs.index(min(diffs))
        return native_sample_rates[min_idx]

    @staticmethod
    def select_best_model_buffer_size(
        io_bs: int, native_buffer_sizes: List[int]
    ) -> int:
        """
        Given a DAW buffer size and a list of all the buffer sizes a Neutone model supports (usually only one, or
        an empty list indicates all buffer sizes are supported), determine the optimal buffer size to use.
        """
        if not native_buffer_sizes:
            return io_bs
        if len(native_buffer_sizes) == 1:
            return native_buffer_sizes[0]
        native_buffer_sizes = sorted(native_buffer_sizes)
        for bs in native_buffer_sizes:
            if bs % io_bs == 0:
                return bs
        for bs in native_buffer_sizes:
            if bs > io_bs:
                return bs
        # TODO(cm): prefer near bs // 2 if 0 padded forward passes are enabled
        # This is a workaround for torchscript not supporting lambda functions
        diffs = [abs(bs - io_bs) for bs in native_buffer_sizes]
        min_idx = diffs.index(min(diffs))
        return native_buffer_sizes[min_idx]

    @staticmethod
    def _calc_saturation_n_case_3_and_4(io_bs: int, model_bs: int) -> int:
        """
        Calculates the saturation n for cases where `io_bs` < `model_bs` or
        `io_bs` > `model_bs` and `io_bs` and `model_bs` do not divide evenly into one
        another. These are the most complicated cases and each line is annotated with
        the intermediate outputs for an example where `io_bs` = 4 and `model_bs` = 7.

        Relevant paper: "Callback Adaptation Techniques" by Stéphane Letz
        (https://hal.science/hal-02158912v1/file/CallbackAdaptation.pdf)
        """
        io_bs_t = tr.tensor(io_bs)  # io_bs_t: tensor(4)
        model_bs_t = tr.tensor(model_bs)  # model_bs_t: tensor(7)
        # Find the LCM of the two buffer sizes
        lcm = tr.lcm(io_bs_t, model_bs_t).item()  # lcm_t: tensor(28)
        # Calculate the remainder samples in the input queue for each step of one cycle.
        # A cycle has a length equal to the number of times we need to push `io_bs`
        # samples onto the input queue and pop `model_bs` samples from the input queue
        # (if possible) such that the input queue will be empty again. Cycle length is
        # just LCM / `io_bs`.
        # remainders: tensor([0, 4, 1, 5, 2, 6, 3])
        remainders = tr.arange(0, lcm, io_bs) % model_bs
        # The maximum remainder is the most important since it represents the largest
        # amount the two queues can be out of sync from each other
        max_remainder = remainders.max()  # max_remainder: tensor(6)
        # Calculate how many `io_bs` buffers are needed to cover the maximum remainder
        # n_io_bufs_in_max_remainder: 2
        n_io_bufs_in_max_remainder = tr.ceil(max_remainder / io_bs).int().item()
        # Calculate the saturation n
        # saturation_n: 12
        saturation_n = io_bs + (n_io_bufs_in_max_remainder * io_bs)
        return saturation_n

    @staticmethod
    def calc_saturation_n(io_bs: int, model_bs: int) -> int:
        """
        Assume you have 2 queues (the input queue and the output queue). Every time
        `io_bs` samples are pushed onto the input queue, the same number of samples must
        be popped from the output queue. Whenever the input queue contains `model_bs`
        samples or more, they are popped from the input queue and pushed onto the output
        queue. This happens instantaneously after pushing to the input queue and before
        popping from the output queue. A simple non-trivial example is when `io_bs` = 4
        and `model_bs` = 7 (saturation_n is 12 not 8 in this case).

        This method calculates the minimum number of samples one must wait before
        popping from the output queue to guarantee that it will never be starved (i.e.
        you cannot pop `io_bs` samples from the output queue). We call this
        `saturation_n` and it will always be a multiple of `io_bs` (since the best case
        scenario is you can pop immediately after the first buffer is pushed onto the
        input queue).
        """
        if model_bs % io_bs == 0:  # Case 1
            return model_bs
        if io_bs % model_bs == 0:  # Case 2
            return io_bs
        # Cases 3 and 4 (`io_bs` < `model_bs` or `io_bs` > `model_bs`)
        return SampleQueueWrapper._calc_saturation_n_case_3_and_4(io_bs, model_bs)

    @staticmethod
    def calc_delay_samples(io_bs: int, model_bs: int) -> int:
        # TODO(cm): document logic behind this
        saturation_n = SampleQueueWrapper.calc_saturation_n(io_bs, model_bs)
        return saturation_n - io_bs

    @staticmethod
    def calc_resampled_buffer_size(orig_sr: int, new_sr: int, orig_bs: int) -> int:
        if orig_sr == new_sr:
            resampled_bs = orig_bs
        else:
            resampled_bs = int(math.ceil(new_sr * orig_bs / orig_sr))
        return resampled_bs

    @staticmethod
    def calc_max_daw_queue_size(
        daw_sr: int, daw_bs: int, model_sr: int, model_bs: int
    ) -> int:
        daw_model_bs = int(model_bs * daw_sr / model_sr) + 1
        return (2 * daw_bs) + daw_model_bs

    def prepare_for_inference(self) -> None:
        self.w2w_base.prepare_for_inference()
        self.use_debug_mode = False
        self.channel_normalizer.use_debug_mode = False
        self.resample_sandwich.use_debug_mode = False
        self.params_resample_sandwich.use_debug_mode = False
        self.in_queue.use_debug_mode = False
        self.params_queue.use_debug_mode = False
        self.out_queue.use_debug_mode = False
        self.eval()

    def _forward(self, resampled_x: Tensor, params: Optional[Tensor] = None) -> None:
        if params is not None:
            params = self.params_resample_sandwich.process_in(params)
            self.params_queue.push(params)

        resampled_in_n = resampled_x.size(1)
        if self.use_debug_mode:
            assert resampled_in_n == self.io_bs

        self.in_queue.push(resampled_x)
        if not self.is_queue_saturated:
            self.seen_n += resampled_in_n
            if self.seen_n >= self.saturation_n:
                self.is_queue_saturated = True

        while self.in_queue.size >= self.model_bs:
            in_popped_n = self.in_queue.pop(self.model_in_buffer)
            if self.use_debug_mode:
                assert in_popped_n == self.model_bs
                validate_waveform(self.model_in_buffer, self.is_input_mono())

            if self.params_queue.is_empty():
                model_out = self.w2w_base.forward(self.model_in_buffer, None)
            else:
                params_popped_n = self.params_queue.pop(self.params_buffer)
                if self.use_debug_mode:
                    assert params_popped_n == in_popped_n
                model_out = self.w2w_base.forward(
                    self.model_in_buffer, self.params_buffer
                )

            if self.use_debug_mode:
                validate_waveform(model_out, self.is_output_mono())
            self.out_queue.push(model_out)

    @tr.jit.export
    def forward(self, x: Tensor, params: Optional[Tensor] = None) -> Tensor:
        is_daw_mono = x.size(0) == 1
        in_n = x.shape[1]
        x = self.channel_normalizer(x, self.is_input_mono(), self.daw_buffer)
        x = self.resample_sandwich.process_in(x)
        self._forward(x, params)

        if self.is_queue_saturated:
            out_popped_n = self.out_queue.pop(self.io_out_buffer)
        else:
            out_popped_n = self.io_out_buffer.size(1)
            self.io_out_buffer.fill_(0)

        x = self.resample_sandwich.process_out(self.io_out_buffer)
        if self.use_debug_mode:
            assert out_popped_n == self.io_out_buffer.size(1)
            assert x.size(1) == in_n
        x = self.channel_normalizer(x, is_daw_mono, self.daw_buffer)
        return x

    @tr.jit.export
    def forward_bt(
        self, x: Tensor, params: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        daw_n_ch = x.size(0)
        is_daw_mono = daw_n_ch == 1
        x = self.channel_normalizer(x, self.is_input_mono(), self.daw_buffer)
        x = self.resample_sandwich.process_in(x)
        self._forward(x, params)

        curr_n = 0
        while self.is_queue_saturated and self.out_queue.size >= self.io_bs:
            out_popped_n = self.out_queue.pop(self.io_out_buffer)
            if self.use_debug_mode:
                assert out_popped_n == self.io_bs
            x = self.resample_sandwich.process_out(self.io_out_buffer)
            x = self.channel_normalizer(x, is_daw_mono, self.daw_buffer)
            if self.use_debug_mode:
                assert x.size(1) == self.daw_bs
                assert curr_n + self.daw_bs <= self.bt_out_buffer.size(1)
            self.bt_out_buffer[0:daw_n_ch, curr_n : curr_n + self.daw_bs] = x
            curr_n += self.daw_bs

        if curr_n == 0:
            return None
        return self.bt_out_buffer[0:daw_n_ch, 0:curr_n]

    @tr.jit.export
    def forward_offline(self, x: Tensor, params: Optional[Tensor] = None) -> Tensor:
        self.reset()

        delay_samples = self.calc_buffering_delay_samples() + self.calc_model_delay_samples()
        if self.use_debug_mode:
            assert x.ndim == 2
            if params is not None:
                assert params.ndim == 2
                assert x.size(1) == params.size(1)

        n_samples = x.size(1)
        # Ensure we pad enough to make up for any delay
        padding_amount = tr.ceil(tr.tensor((n_samples + delay_samples) / self.daw_bs)) * self.daw_bs - n_samples
        padding_amount = int(padding_amount.item())
        padded_audio = tr.nn.functional.pad(x, [0, padding_amount])
        audio_chunks = padded_audio.split(self.daw_bs, dim=1)

        param_chunks = []
        if params is not None:
            padded_params = tr.nn.functional.pad(params, [0, padding_amount], mode="replicate")
            param_chunks = padded_params.split(self.daw_bs, dim=1)

        # TODO(cm): if memory is an issue, we can preallocate everything beforehand
        out_chunks = []
        for idx, audio_chunk in enumerate(audio_chunks):
            if params is None:
                out = self.forward(audio_chunk, params=None).clone()
            else:
                param_chunk = param_chunks[idx]
                out = self.forward(audio_chunk, params=param_chunk).clone()
            out_chunks.append(out)

        audio_out = tr.cat(out_chunks, dim=1)
        audio_out = audio_out[:, -n_samples:]
        return audio_out

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return self.w2w_base.is_input_mono()

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return self.w2w_base.is_output_mono()

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return self.w2w_base.get_native_sample_rates()

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return self.w2w_base.get_native_buffer_sizes()

    @tr.jit.export
    def is_resampling(self) -> bool:
        return self.resample_sandwich.is_resampling()

    @tr.jit.export
    def calc_buffering_delay_samples(self) -> int:
        delay_samples = self.calc_delay_samples(self.io_bs, self.model_bs)
        if self.is_resampling():
            delay_samples = int(delay_samples * self.daw_bs / self.io_bs)
        return delay_samples

    @tr.jit.export
    def calc_model_delay_samples(self) -> int:
        delay_samples = self.w2w_base.calc_model_delay_samples()
        if self.is_resampling():
            delay_samples = int(delay_samples * self.daw_bs / self.io_bs)
        return delay_samples

    @tr.jit.export
    def set_daw_sample_rate_and_buffer_size(
        self,
        daw_sr: int,
        daw_bs: int,
        model_sr: Optional[int] = None,
        model_bs: Optional[int] = None,
    ) -> int:
        # Sample rate
        if model_sr is not None:
            if self.use_debug_mode:
                assert (
                    len(self.get_native_sample_rates()) == 0
                    or model_sr in self.get_native_sample_rates()
                )
        else:
            model_sr = self.select_best_model_sr(daw_sr, self.get_native_sample_rates())

        io_bs = self.calc_resampled_buffer_size(daw_sr, model_sr, daw_bs)

        self.resample_sandwich.set_sample_rates(daw_sr, model_sr, daw_bs)
        self.params_resample_sandwich.set_sample_rates(daw_sr, model_sr, daw_bs)
        self.daw_sr = daw_sr
        self.model_sr = model_sr

        # Buffer size
        if model_bs is not None:
            if self.use_debug_mode:
                assert (
                    len(self.get_native_buffer_sizes()) == 0
                    or model_bs in self.get_native_buffer_sizes()
                )
        else:
            model_bs = self.select_best_model_buffer_size(
                io_bs, self.get_native_buffer_sizes()
            )

        self.w2w_base.set_sample_rate_and_buffer_size(model_sr, model_bs)
        self.daw_bs = daw_bs
        self.io_bs = io_bs
        self.model_bs = model_bs

        self.in_queue = CircularInplaceTensorQueue(
            self.in_n_ch,
            (2 * self.io_bs) + self.model_bs,
            use_debug_mode=self.use_debug_mode,
        )
        self.params_queue = CircularInplaceTensorQueue(
            self.w2w_base.MAX_N_PARAMS,
            (2 * self.io_bs) + self.model_bs,
            use_debug_mode=self.use_debug_mode,
        )
        self.out_queue = CircularInplaceTensorQueue(
            self.out_n_ch,
            (2 * self.io_bs) + self.model_bs,
            use_debug_mode=self.use_debug_mode,
        )

        self.daw_buffer = tr.zeros((2, self.daw_bs))
        self.model_in_buffer = tr.zeros((self.in_n_ch, self.model_bs))
        self.params_buffer = tr.zeros((self.w2w_base.MAX_N_PARAMS, self.model_bs))
        self.io_out_buffer = tr.zeros((self.out_n_ch, self.io_bs))

        max_daw_queue_size = self.calc_max_daw_queue_size(
            self.daw_sr, self.daw_bs, self.model_sr, self.model_bs
        )
        self.bt_out_buffer = tr.zeros((2, max_daw_queue_size))

        self.saturation_n = self.calc_saturation_n(self.io_bs, self.model_bs)
        self.reset()

        return max_daw_queue_size

    @tr.jit.export
    def reset(self) -> None:
        self.w2w_base.reset()
        self.in_queue.reset()
        self.params_queue.reset()
        self.out_queue.reset()
        self.daw_buffer.fill_(0)
        self.model_in_buffer.fill_(0)
        self.params_buffer.fill_(0)
        self.io_out_buffer.fill_(0)
        self.bt_out_buffer.fill_(0)
        self.seen_n = 0
        self.is_queue_saturated = False

    @tr.jit.export
    def get_preserved_attributes(self) -> List[str]:
        return [
            "forward_bt",
            "is_input_mono",
            "is_output_mono",
            "get_model_name",
            "get_model_authors",
            "get_native_sample_rates",
            "get_native_buffer_sizes",
            "get_wet_default_value",
            "get_dry_default_value",
            "get_default_param_values",
            "get_input_gain_default_value",
            "get_output_gain_default_value",
            "is_resampling",
            "calc_buffering_delay_samples",
            "calc_model_delay_samples",
            "set_daw_sample_rate_and_buffer_size",
            "reset",
            "get_preserved_attributes",
            "to_metadata",
            "w2w_base",
        ]

    @tr.jit.export
    def to_metadata(self) -> WaveformToWaveformMetadata:
        return self.w2w_base.to_metadata()

    @tr.jit.export
    def get_model_name(self) -> str:
        return self.w2w_base.get_model_name()

    @tr.jit.export
    def get_model_authors(self) -> List[str]:
        return self.w2w_base.get_model_authors()

    @tr.jit.export
    def get_wet_default_value(self) -> float:
        return self.w2w_base.get_wet_default_value()

    @tr.jit.export
    def get_dry_default_value(self) -> float:
        return self.w2w_base.get_dry_default_value()

    @tr.jit.export
    def get_default_param_values(self) -> Tensor:
        return self.w2w_base.get_default_param_values()

    @tr.jit.export
    def get_default_param_names(self) -> List[str]:
        return self.w2w_base.get_default_param_names()

    @tr.jit.export
    def get_default_param_descriptions(self) -> List[str]:
        return self.w2w_base.get_default_param_descriptions()

    @tr.jit.export
    def get_default_param_types(self) -> List[str]:
        return self.w2w_base.get_default_param_types()

    @tr.jit.export
    def get_default_param_used(self) -> List[bool]:
        return self.w2w_base.get_default_param_used()

    @tr.jit.export
    def get_input_gain_default_value(self) -> float:
        return self.w2w_base.get_input_gain_default_value()

    @tr.jit.export
    def get_output_gain_default_value(self) -> float:
        return self.w2w_base.get_output_gain_default_value()
