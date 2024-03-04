import logging
import os
import pathlib
from argparse import ArgumentParser
from typing import Dict, List

import torch as tr
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter, KnobNeutoneParameter
from neutone_sdk.realtime_stft import RealtimeSTFT
from neutone_sdk.utils import save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SpectralFilter(nn.Module):
    def __init__(self) -> None:
        """
        Creates a spectral notch filter, where the bandwidth of the filter also changes as the center frequency changes.
        """
        super().__init__()
        self.base_constant = tr.tensor(
            1025 / tr.e
        )  # Used to scale the controls somewhat to the STFT
        self.half_constant = tr.tensor(0.5)  # Prevent dynamic memory allocations

    def _map_0to1_val_to_log_bin_idx(self, val: Tensor, max_bin: int) -> int:
        """
        Maps a float tensor between [0.0, 1.0] to an integer between [0, max_bins] with the assumption that the
        bin indices follow a logarithmic spacing.
        """
        idx = (
            (tr.pow(self.base_constant, val) - 1.0)
            / (self.base_constant - 1.0)
            * max_bin
        )
        idx = int(tr.clip(tr.round(idx), 0, max_bin))
        return idx

    def forward(
        self, x: Tensor, center: Tensor, width: Tensor, amount: Tensor
    ) -> Tensor:
        """
        Filters a positive valued magnitude spectrogram using a notch filter with controllable center, width,
        and amount of attenuation.

        Args:
            x: a magnitude spectrogram with shape (n_ch, n_bins, n_frames)
            center: 1D control value between [0.0, 1.0] for the center frequency of the filter.
            width: 1D control value between [0.0, 1.0] for the bandwidth of the filter.
            amount: 1D control value between [0.0, 1.0] for the amount of attenuation.
        """
        if amount == 0.0:
            return x
        n_bins = x.size(1)  # Figure out how many bins we have to work with
        # Find the center freq bin
        center_bin_idx = self._map_0to1_val_to_log_bin_idx(center, n_bins - 1)
        # Find the lowest freq bin
        lo_bin_idx = self._map_0to1_val_to_log_bin_idx(
            center * (1.0 - width), n_bins - 1
        )
        lo_bin_idx = max(0, lo_bin_idx)
        # Find the highest freq bin
        hi_bin_idx = self._map_0to1_val_to_log_bin_idx(
            center + ((1.0 - center) * width), n_bins - 1
        )
        hi_bin_idx = min(n_bins - 1, hi_bin_idx)
        # If the filter has 0 width, we don't need to do anything
        if hi_bin_idx - lo_bin_idx == 0:
            return x
        # Filter the low bins of the notch
        if center_bin_idx - lo_bin_idx > 0:
            # Using a linear spacing here is not ideal since the frequency bins are not linearly spaced,
            # but this is just an example
            lo_filter = 1.0 - (
                tr.linspace(0.0, 1.0, center_bin_idx - lo_bin_idx + 2)[1:-1] * amount
            )
            lo_filter = lo_filter.view(1, -1, 1)
            x[:, lo_bin_idx:center_bin_idx, :] *= lo_filter
        # Filter the high bins of the notch
        if hi_bin_idx - center_bin_idx > 0:
            # Using a linear spacing here is not ideal since the frequency bins are not linearly spaced,
            # but this is just an example
            hi_filter = 1.0 - (
                tr.linspace(1.0, 0.0, hi_bin_idx - center_bin_idx + 1)[:-1] * amount
            )
            hi_filter = hi_filter.view(1, -1, 1)
            x[:, center_bin_idx:hi_bin_idx, :] *= hi_filter
        return x


class SpectralFilterWrapper(WaveformToWaveformBase):
    def __init__(
        self,
        spectral_filter_model: nn.Module,
        model_io_n_frames: int = 16,
        n_fft: int = 2048,
        hop_len: int = 512,
        fade_n_samples: int = 384,  # Cross-fade for 3/4 of the hop_len to ensure no buzzing in the wet audio
        use_debug_mode: bool = True,
    ) -> None:
        """
        Creates a modified WaveformToWaveformBase wrapper that can be used to create spectral neural audio effects.
        Feel free to use this as a starting point to make your own spectral effects!

        Args:
            spectral_filter_model: a spectral model, in this example a filter (could be replaced with anything).
            model_io_n_frames: the number of STFT frames the spectral model expects as input and output.
            n_fft: n_fft to use for the STFT.
            hop_len: hop_len in samples to use for the STFT.
            fade_n_samples: no. of samples to crossfade between output buffers of audio after the inverse STFT. Adds a
                            slight delay, but prevents clicks and pops in the output audio.
            use_debug_mode: makes debugging easier, is turned off automatically before the model is exported.
        """
        super().__init__(spectral_filter_model, use_debug_mode)
        in_ch = 1 if self.is_input_mono() else 2
        self.stft = RealtimeSTFT(
            model_io_n_frames=model_io_n_frames,
            io_n_ch=in_ch,
            n_fft=n_fft,
            hop_len=hop_len,
            power=1.0,  # Ensures an energy spectrogram
            logarithmize=False,  # We don't need a log-magnitude spectrogram for this filter
            ensure_pos_spec=True,  # Ensures a positive-valued spectrogram
            use_phase_info=True,  # Keep the phase information for the inverse STFT
            fade_n_samples=fade_n_samples,
            use_debug_mode=use_debug_mode,
        )
        self.stft.set_buffer_size(self.stft.calc_min_buffer_size())
        if use_debug_mode:
            log.info(f"Supported buffer sizes = {self.get_native_buffer_sizes()}")
            log.info(f"Supported sample rate = {self.get_native_sample_rates()}")
            log.info(f"STFT delay = {self.calc_model_delay_samples()}")

    def get_model_name(self) -> str:
        return "spectral.filter"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "Spectral notch filter."

    def get_model_long_description(self) -> str:
        return (
            "Filters the audio in the spectral domain using a central frequency, bandwidth, and amount. "
            "The bandwidth changes as the central frequency changes."
        )

    def get_technical_description(self) -> str:
        return (
            "Filters the audio in the spectral domain using a central frequency, bandwidth, and amount. "
            "The bandwidth changes as the central frequency changes."
        )

    def get_technical_links(self) -> Dict[str, str]:
        return {}

    def get_tags(self) -> List[str]:
        return ["spectral", "filter", "notch filter", "stft", "template"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            KnobNeutoneParameter(
                "center", "center frequency of the filter", default_value=0.3
            ),
            KnobNeutoneParameter("width", "width of the filter", default_value=0.5),
            KnobNeutoneParameter(
                "amount", "spectral attenuation amount", default_value=0.9
            ),
        ]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return False

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return False

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        # For consistent filtering across different sampling rates, a native sampling rate must be given. Feel free to
        # change this to your required sampling rate.
        return [44100]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return (
            self.stft.calc_supported_buffer_sizes()
        )  # Possible buffer sizes are determined by the STFT parameters

    @tr.jit.export
    def calc_model_delay_samples(self) -> int:
        # TODO(cm): make a model specific version of this method?
        return self.stft.calc_model_delay_samples()  # This is equal to `fade_n_samples`

    def set_model_buffer_size(self, n_samples: int) -> bool:
        self.stft.set_buffer_size(n_samples)
        return True

    def reset_model(self) -> bool:
        self.stft.reset()
        return True

    def prepare_for_inference(self) -> None:
        super().prepare_for_inference()
        # This needs to be done explicitly until we have dedicated wrapper base class for spectral models
        self.stft.use_debug_mode = False
        self.stft.eval()

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        center, width, amount = params["center"], params["width"], params["amount"]
        x = self.stft.audio_to_spec(
            x
        )  # Convert the audio to a spectrogram (n_ch, n_bins, n_frames)
        x = self.model.forward(
            x, center, width, amount
        )  # Apply the spectral filter and receive an altered spectrogram
        x = self.stft.spec_to_audio(
            x
        )  # Convert the filtered spectrogram back to audio (n_ch, n_samples)
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    root_dir = pathlib.Path(args.output)

    model = SpectralFilter()
    wrapper = SpectralFilterWrapper(model)
    save_neutone_model(wrapper, root_dir, dump_samples=True, submission=True)
