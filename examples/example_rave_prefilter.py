import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
from torch import Tensor, nn

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.audio import (
    AudioSample,
    AudioSamplePair,
    render_audio_sample,
)
from neutone_sdk.filters import FIRFilter, FilterType
from neutone_sdk.utils import save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class FilteredRAVEModelWrapper(WaveformToWaveformBase):
    def __init__(self, model: nn.Module, use_debug_mode: bool = True) -> None:
        super().__init__(model, use_debug_mode)
        # filter to be applied before model
        # cut below 500 and above 4000 Hz
        self.pre_filter = FIRFilter(
            FilterType.BANDPASS, cutoffs=[500.0, 4000.0], filt_size=257
        )

    def get_model_name(self) -> str:
        return "RAVE.example"  # <-EDIT THIS

    def get_model_authors(self) -> List[str]:
        return ["Author Name"]  # <-EDIT THIS

    def get_model_short_description(self) -> str:
        return "RAVE model trained on xxx sounds."  # <-EDIT THIS

    def get_model_long_description(self) -> str:
        return (  # <-EDIT THIS
            "RAVE timbre transfer model trained on xxx sounds. Useful for xxx sounds."
        )

    def get_technical_description(self) -> str:
        return "RAVE model proposed by Caillon, Antoine et al."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2111.05011",
            "Code": "https://github.com/acids-ircam/RAVE",
        }

    def get_tags(self) -> List[str]:
        return ["timbre transfer", "RAVE"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        """
        set to True for models in experimental stage
        (status shown on the website)
        """
        return True  # <-EDIT THIS

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter(
                name="Chaos", description="Magnitude of latent noise", default_value=0.0
            ),
            NeutoneParameter(
                name="Z edit index",
                description="Index of latent dimension to edit",
                default_value=0.0,
            ),
            NeutoneParameter(
                name="Z scale",
                description="Scale of latent variable",
                default_value=0.5,
            ),
            NeutoneParameter(
                name="Z offset",
                description="Offset of latent variable",
                default_value=0.5,
            ),
        ]

    def is_input_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    def is_output_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    def get_native_sample_rates(self) -> List[int]:
        return [48000]  # <-Set to model sr during training

    def get_native_buffer_sizes(self) -> List[int]:
        return [2048]

    def calc_model_delay_samples(self) -> int:
        # model latency should also be added if non-causal
        return self.pre_filter.delay + 2048

    def set_model_sample_rate_and_buffer_size(
        self, sample_rate: int, n_samples: int
    ) -> bool:
        # Set prefilter samplerate to current sample rate
        self.pre_filter.set_parameters(sample_rate=sample_rate)
        return True

    def get_citation(self) -> str:
        return """Caillon, A., & Esling, P. (2021). RAVE: A variational autoencoder for fast and high-quality neural audio synthesis. arXiv preprint arXiv:2111.05011."""

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # Apply pre-filter
        x = self.pre_filter(x)
        # parameters edit the latent variable
        z = self.model.encode(x.unsqueeze(1))
        noise_amp = params["Chaos"] * 2
        z = torch.randn_like(z) * noise_amp + z
        # add offset / scale
        idx_z = int(
            torch.clamp(params["Z edit index"], min=0.0, max=0.99)
            * self.model.latent_size
        )
        z_scale = params["Z scale"] * 2  # 0~1 -> 0~2
        z_offset = params["Z offset"] * 2 - 1  # 0~1 -> -1~1
        z[:, idx_z] = z[:, idx_z] * z_scale + z_offset
        out = self.model.decode(z)
        out = out.squeeze(1)
        return out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="./models/rave/rave_cached.ts",
        help="exported RAVE torchscript file",
    )
    parser.add_argument("-o", "--output", default="ravemodel", help="model output name")
    parser.add_argument("-f", "--folder", default="./exports", help="output folder")
    parser.add_argument(
        "-s",
        "--sounds",
        nargs="*",
        type=str,
        default=None,
        help="directory of sounds to use as example input.",
    )
    args = parser.parse_args()
    root_dir = Path(args.folder) / args.output

    # wrap it
    model = torch.jit.load(args.input)
    wrapper = FilteredRAVEModelWrapper(model)

    soundpairs = None
    if args.sounds is not None:
        soundpairs = []
        for sound in args.sounds:
            wave, sr = torchaudio.load(sound)
            input_sample = AudioSample(wave, sr)
            rendered_sample = render_audio_sample(wrapper, input_sample)
            soundpairs.append(AudioSamplePair(input_sample, rendered_sample))

    save_neutone_model(
        wrapper,
        root_dir,
        freeze=False,
        dump_samples=True,
        submission=True,
        audio_sample_pairs=soundpairs,
    )
