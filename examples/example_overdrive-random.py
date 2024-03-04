# This code is based on the following repository written by Christian J. Steinmetz
# https://github.com/csteinmetz1/micro-tcn
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter, KnobNeutoneParameter
from neutone_sdk.tcn_1d import FiLM
from neutone_sdk.utils import save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


# TODO(christhetree): integrate this into tcn_1d.py
class PaddingCached(nn.Module):  # to maintain signal continuity over sample windows
    def __init__(self, padding: int, channels: int) -> None:
        super().__init__()
        self.padding = padding
        self.channels = channels
        pad = torch.zeros(1, self.channels, self.padding)
        self.register_buffer("pad", pad)

    def forward(self, x: Tensor) -> Tensor:
        padded_x = torch.cat([self.pad, x], -1)  # concat input signal to the cache
        self.pad = padded_x[..., -self.padding :]  # discard old cache
        return padded_x


# TODO(christhetree): integrate this into tcn_1d.py
class Conv1dCached(nn.Module):  # Conv1d with cache
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int = 1,
        weight_norm: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.pad = PaddingCached(padding * 2, in_chan)
        self.conv = nn.Conv1d(
            in_chan, out_chan, kernel, stride, dilation=dilation, bias=bias
        )
        nn.init.normal_(self.conv.weight)  # random initialization
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)  # get (cached input + current input)
        x = self.conv(x)
        return x


# TODO(christhetree): integrate this into tcn_1d.py
class TCNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        cond_dim: int = 32,
    ) -> None:
        super(TCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        padding = kernel_size // 2 * dilation
        self.conv1 = Conv1dCached(
            in_ch,
            out_ch,
            kernel=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.res = nn.Conv1d(
            in_ch, out_ch, kernel_size=1, groups=1, bias=False
        )  # residual connection
        self.bn = nn.BatchNorm1d(out_ch)
        self.film = FiLM(out_ch, cond_dim)
        self.relu = nn.PReLU(out_ch)

    def forward(self, x: Tensor, p: Tensor) -> Tensor:
        x_in = x
        x = self.conv1(x)
        x = self.film(x, p)
        x = self.bn(x)
        x = self.relu(x)

        # residual
        x_res = self.res(x_in)
        start = (x_res.shape[-1] - x.shape[-1]) // 2
        stop = start + x.shape[-1]
        x = x + x_res[..., start:stop]
        return x


class OverdriveModel(nn.Module):
    def __init__(
        self,
        ninputs: int = 1,
        noutputs: int = 1,
        nblocks: int = 4,
        channel_growth: int = 0,
        channel_width: int = 32,
        kernel_size: int = 13,
        dilation_growth: int = 2,
        ncondition: int = 2,
    ) -> None:
        super().__init__()

        # MLP layers for conditioning
        self.ncondition = ncondition
        self.condition = torch.nn.Sequential(
            torch.nn.Linear(ncondition, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),  # cond_dim = 32
            torch.nn.ReLU(),
        )

        # main model
        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            out_ch = in_ch * channel_growth if channel_growth > 1 else channel_width
            dilation = dilation_growth**n
            self.blocks.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation, cond_dim=32)
            )
        self.output = nn.Conv1d(out_ch, noutputs, kernel_size=1)

        # random initialization
        self.initialize_random()

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        p = self.condition(c)  # conditioning
        for _, block in enumerate(self.blocks):
            x = block(x, p)
        y = torch.tanh(self.output(x))  # clipping
        return y

    def weights_init(self, m: nn.Module) -> None:
        classname = m.__class__.__name__
        if classname == "Linear":
            nn.init.normal_(m.weight, 0, 0.40)

    def initialize_random(self) -> None:
        for n in self.blocks:
            nn.init.normal_(n.conv1.conv.weight, 0, 0.7)
            # nn.init.normal_(self.output.weight, 0, 0.25)
        self.condition.apply(self.weights_init)


class OverdriveModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "conv1d-overdrive.random"

    def get_model_authors(self) -> List[str]:
        return ["Nao Tokui"]

    def get_model_short_description(self) -> str:
        return "Neural distortion/overdrive effect"

    def get_model_long_description(self) -> str:
        return "Neural distortion/overdrive effect through randomly initialized Convolutional Neural Network"

    def get_technical_description(self) -> str:
        return "Random distortion/overdrive effect through randomly initialized Temporal-1D-convolution layers. Based on the idea proposed by Steinmetz et al."

    def get_tags(self) -> List[str]:
        return ["distortion", "overdrive"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2010.04237",
            "Code": "https://github.com/csteinmetz1/micro-tcn",
        }

    def get_citation(self) -> str:
        return "Steinmetz, C. J., & Reiss, J. D. (2020). Randomized overdrive neural networks. arXiv preprint arXiv:2010.04237."

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            KnobNeutoneParameter("depth", "Effect Depth", 0.0),
            KnobNeutoneParameter("P1", "Feature modulation 1", 0.0),
            KnobNeutoneParameter("P2", "Feature modulation 2", 0.0),
        ]

    @torch.jit.export
    def is_input_mono(self) -> bool:
        return False

    @torch.jit.export
    def is_output_mono(self) -> bool:
        return False

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # conditioning for FiLM layer
        p1 = params["P1"]
        p2 = params["P2"]
        depth = params["depth"]
        condition = torch.hstack([p1, p2]).reshape((1, -1)) * depth

        # main process
        for ch in range(x.shape[0]):  # process channel by channel
            x_ = x[ch].reshape(1, 1, -1)
            x_ = self.model(x_, condition)
            x[ch] = x_.squeeze()
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    root_dir = Path(args.output)

    model = OverdriveModel()
    wrapper = OverdriveModelWrapper(model)
    metadata = wrapper.to_metadata()
    save_neutone_model(wrapper, root_dir, dump_samples=True, submission=True)
