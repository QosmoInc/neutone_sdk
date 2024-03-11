import base64
from cffi import FFI
from dataclasses import dataclass
import logging
import math
import io
import pkgutil
from typing import Optional, List, Union
from typing_extensions import Self

import numpy as np
import torch as tr
from torch import nn, Tensor
import torchaudio
import soundfile as sf
from torch.jit import ScriptModule
from tqdm import tqdm

import neutone_sdk

logging.basicConfig()
log = logging.getLogger(__name__)


def write_mp3(buffer: io.BytesIO, y: tr.Tensor, sr: int, quality: float = 0):
    """
    We're using this instead of sf.write in order to change the bitrate,
    where quality goes from 0 (high) to 1 (low).

    The API is similar to torchaudio.save, so y should be (num_channels, num_samples).
    """
    assert 0 <= quality <= 1
    assert (
        y.shape[0] < y.shape[1]
    ), "Expecting  audio to have a shape of (num_channels, num_samples), try swapping the dimensions"
    ffi = FFI()
    quality = ffi.new("double *")
    vbr_set = ffi.new("int *")
    with sf.SoundFile(
        buffer, "w", channels=y.shape[0], samplerate=sr, format="mp3"
    ) as f:
        quality[0] = 0  # 0[high]~1[low]
        # 0x1301 - SFC_SET_COMPRESSION_LEVEL
        c = sf._snd.sf_command(f._file, 0x1301, quality, 8)
        assert c == sf._snd.SF_TRUE, "Couldn't set bitrate on MP3"

        # 0x1305 - SFC_SET_BITRATE_MODE
        vbr_set[0] = 2  # 0 - CONSTANT, 1 - AVERAGE, 2 - VARIABLE
        c = sf._snd.sf_command(f._file, 0x1305, vbr_set, 4)
        assert c == sf._snd.SF_TRUE, "Couldn't set MP3 to VBR"

        f.write(y.T.numpy())
    assert f.closed


@dataclass
class AudioSample:
    """
    AudioSample is simply a pair of (audio, sample_rate) that is easier to work
    with within the SDK. We recommend users to read and write to mp3 files as
    they are better supported and formats like ogg can have subtle bugs when
    reading and writing using the current backend (soundfile).
    """

    audio: Tensor
    sr: int

    def __post_init__(self):
        assert self.audio.ndim == 2
        assert (
            self.audio.size(0) == 1 or self.audio.size(0) == 2
        ), "Audio sample audio should be 1 or 2 channels, channels first"

    def is_mono(self) -> bool:
        return self.audio.size(0) == 1

    def to_mp3_bytes(self) -> bytes:
        buff = io.BytesIO()
        write_mp3(buff, self.audio, self.sr)
        buff.seek(0)
        return buff.read()

    def to_mp3_b64(self) -> str:
        return base64.b64encode(self.to_mp3_bytes()).decode()

    @classmethod
    def from_bytes(cls, bytes_: bytes) -> Self:
        y, sr = sf.read(io.BytesIO(bytes_), always_2d=True)
        return cls(tr.from_numpy(y.T.astype(np.float32)), sr)

    @classmethod
    def from_file(cls, path: str) -> Self:
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())

    @classmethod
    def from_b64(cls, b64_sample: str) -> Self:
        return cls.from_bytes(base64.b64decode(b64_sample))


@dataclass
class AudioSamplePair:
    input: AudioSample
    output: AudioSample

    def to_metadata_format(self):
        return {
            "in": self.input.to_mp3_b64(),
            "out": self.output.to_mp3_b64(),
        }


def get_default_audio_samples() -> List[AudioSample]:
    """
    Returns a list of audio samples to be displayed on the website.

    The SDK provides one sample by default, but this method can be used to
    provide different samples.

    By default the outputs of this function will be ran through the model
    and the prerendered samples will be stored inside the saved object.

    See get_prerendered_audio_samples and render_audio_sample for more details.
    """
    log.info(
        "Using default sample... Please consider using your own audio samples by overriding the get_audio_samples method"
    )
    sample_ambience = AudioSample.from_bytes(
        pkgutil.get_data(__package__, "assets/default_samples/sample_ambience.mp3"),
    )
    sample_drums = AudioSample.from_bytes(
        pkgutil.get_data(__package__, "assets/default_samples/sample_drums.mp3"),
    )
    sample_rhodes = AudioSample.from_bytes(
        pkgutil.get_data(__package__, "assets/default_samples/sample_rhodes.mp3"),
    )

    return [sample_rhodes, sample_drums, sample_ambience]


def render_audio_sample(
    model: Union["SampleQueueWrapper", "WaveformToWaveformBase", ScriptModule],
    input_sample: AudioSample,
    params: Optional[Tensor] = None,
    output_sr: int = 44100,
) -> AudioSample:
    """
    params: either [model.MAX_N_PARAMS] 1d tensor of constant parameter values
            or [model.MAX_N_PARAMS, input_sample.audio.size(1)] 2d tensor of parameter values for every input audio sample
    """

    with tr.no_grad():
        model.use_debug_mode = True  # Turn on debug mode to catch common mistakes when rendering sample audio

        preferred_sr = neutone_sdk.SampleQueueWrapper.select_best_model_sr(
            input_sample.sr, model.get_native_sample_rates()
        )
        if len(model.get_native_buffer_sizes()) > 0:
            buffer_size = model.get_native_buffer_sizes()[0]
        else:
            buffer_size = 512

        audio = input_sample.audio
        if input_sample.sr != preferred_sr:
            audio = torchaudio.transforms.Resample(input_sample.sr, preferred_sr)(audio)

        if model.is_input_mono() and not input_sample.is_mono():
            audio = tr.mean(audio, dim=0, keepdim=True)
        elif not model.is_input_mono() and input_sample.is_mono():
            audio = audio.repeat(2, 1)

        audio_len = audio.size(1)
        padding_amount = math.ceil(audio_len / buffer_size) * buffer_size - audio_len
        padded_audio = nn.functional.pad(audio, [0, padding_amount])
        audio_chunks = padded_audio.split(buffer_size, dim=1)

        model.set_daw_sample_rate_and_buffer_size(
            preferred_sr, buffer_size, preferred_sr, buffer_size
        )

        # make sure the shape of params is compatible with the model calls.
        if params is not None:
            assert params.shape[0] == model.MAX_N_PARAMS

            # if constant values, copy across audio dimension
            if params.dim() == 1:
                params = params.repeat([audio_len, 1]).T

            # otherwise resample to match audio
            else:
                assert params.shape == (model.MAX_N_PARAMS, input_sample.audio.size(1))
                params = torchaudio.transforms.Resample(input_sample.sr, preferred_sr)(
                    params
                )
                params = tr.clamp(params, 0, 1)

            # padding and chunking parameters to match audio
            padded_params = nn.functional.pad(
                params, [0, padding_amount], mode="replicate"
            )
            param_chunks = padded_params.split(buffer_size, dim=1)

            out_chunks = [
                model.forward(audio_chunk, param_chunk).clone()
                for audio_chunk, param_chunk in tqdm(
                    zip(audio_chunks, param_chunks), total=len(audio_chunks)
                )
            ]

        else:
            out_chunks = [
                model.forward(audio_chunk, None).clone()
                for audio_chunk in tqdm(audio_chunks)
            ]

        audio_out = tr.hstack(out_chunks)[:, :audio_len]

        model.reset()

        if preferred_sr != output_sr:
            audio_out = torchaudio.transforms.Resample(preferred_sr, output_sr)(
                audio_out
            )

        # Make the output audio consistent with the input audio
        if audio_out.size(0) == 1 and not input_sample.is_mono():
            audio_out = audio_out.repeat(2, 1)
        elif audio_out.size(0) == 2 and input_sample.is_mono():
            audio_out = tr.mean(audio_out, dim=0, keepdim=True)

        return AudioSample(audio_out, output_sr)
