import logging
import os
import pathlib
from argparse import ArgumentParser
from typing import Dict, List

import torch as tr
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class DelayedPassthroughModel(nn.Module):
    def __init__(self, delay_n_samples: int, in_ch: int = 2) -> None:
        super().__init__()
        self.delay_n_samples = delay_n_samples
        self.delay_buf = tr.zeros((in_ch, delay_n_samples))

    def forward(self, x: Tensor) -> Tensor:
        x = tr.cat([self.delay_buf, x], dim=-1)
        self.delay_buf[:, :] = x[:, -self.delay_n_samples :]
        x = x[:, : -self.delay_n_samples]
        return x


class DelayedPassthroughModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "delayed.passthrough"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "Delayed passthrough model."

    def get_model_long_description(self) -> str:
        return "Delays the input audio by some number of samples. Should be tested with 50/50 dry/wet."

    def get_technical_description(self) -> str:
        return "Delays the input audio by some number of samples. Should be tested with 50/50 dry/wet."

    def get_technical_links(self) -> Dict[str, str]:
        return {}

    def get_tags(self) -> List[str]:
        return []

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return []

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return False

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return False

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [44100]  # Change this to test different scenarios

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return [2048]  # Change this to test different scenarios

    @tr.jit.export
    def reset_model(self) -> bool:
        self.model.delay_buf.fill_(0)
        return True

    @tr.jit.export
    def calc_model_delay_samples(self) -> int:
        return self.model.delay_n_samples

    @tr.jit.export
    def get_wet_default_value(self) -> float:
        return 0.5

    @tr.jit.export
    def get_dry_default_value(self) -> float:
        return 0.5

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        x = self.model.forward(x)
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    root_dir = pathlib.Path(args.output)

    model = DelayedPassthroughModel(
        delay_n_samples=500
    )  # Change delay_n_samples to test different scenarios
    wrapper = DelayedPassthroughModelWrapper(model)
    save_neutone_model(wrapper, root_dir, dump_samples=True, submission=True)
