import logging
import os
import pathlib
from argparse import ArgumentParser
from typing import Dict, List

import torch as tr
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter, ContinuousNeutoneParameter
from neutone_sdk.utils import save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ClipperModel(nn.Module):
    def forward(
        self, x: Tensor, min_val: Tensor, max_val: Tensor, gain: Tensor
    ) -> Tensor:
        tr.neg(min_val, out=min_val)
        tr.mul(gain, min_val, out=min_val)
        tr.mul(gain, max_val, out=max_val)
        tr.clip(x, min=min_val, max=max_val, out=x)
        return x


class ClipperModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "clipper"

    def get_model_authors(self) -> List[str]:
        return ["Andrew Fyfe"]

    def get_model_short_description(self) -> str:
        return "Audio clipper."

    def get_model_long_description(self) -> str:
        return "Clips the input audio between -1 and 1."

    def get_technical_description(self) -> str:
        return "Clips the input audio between -1 and 1."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Code": "https://github.com/QosmoInc/neutone_sdk/blob/main/examples/example_clipper.py"
        }

    def get_tags(self) -> List[str]:
        return ["clipper"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            ContinuousNeutoneParameter("min", "min clip threshold", default_value=0.15),
            ContinuousNeutoneParameter("max", "max clip threshold", default_value=0.15),
            ContinuousNeutoneParameter("gain", "scale clip threshold", default_value=1.0),
        ]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return False

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return False

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def aggregate_params(self, params: Tensor) -> Tensor:
        return params  # We want sample-level control, so no aggregation

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        min_val, max_val, gain = params["min"], params["max"], params["gain"]
        x = self.model.forward(x, min_val, max_val, gain)
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    root_dir = pathlib.Path(args.output)

    model = ClipperModel()
    wrapper = ClipperModelWrapper(model)
    save_neutone_model(wrapper, root_dir, dump_samples=True, submission=True)
