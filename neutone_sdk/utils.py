import io
import json
import logging
import os
import random
from pathlib import Path
from typing import Tuple, Dict, List

import torch as tr
from torch import Tensor
from torch.jit import ScriptModule

from neutone_sdk.audio import (
    AudioSample,
    AudioSamplePair,
    get_default_audio_samples,
    render_audio_sample,
)
from neutone_sdk.constants import MAX_N_AUDIO_SAMPLES, MAX_N_PARAMS
from neutone_sdk.core import NeutoneModel
from neutone_sdk.metadata import validate_metadata

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def dump_samples_from_metadata(metadata: Dict, root_dir: Path) -> None:
    log.info(f"Dumping samples to {root_dir/'samples'}...")
    os.makedirs(root_dir / "samples", exist_ok=True)
    for i, sample in enumerate(metadata["sample_sound_files"]):
        with open(root_dir / "samples" / f"sample_in_{i}.mp3", "wb") as f:
            f.write(AudioSample.from_b64(sample["in"]).to_mp3_bytes())
        with open(root_dir / "samples" / f"sample_out_{i}.mp3", "wb") as f:
            f.write(AudioSample.from_b64(sample["out"]).to_mp3_bytes())


def model_to_torchscript(
    model: "WaveformToWaveformBase",
    freeze: bool = False,
    optimize: bool = False,
) -> ScriptModule:
    model.prepare_for_inference()
    script = tr.jit.script(model)
    check_for_preserved_attributes(script, script.get_preserved_attributes())
    if freeze:
        log.warning(
            f"Freezing is not recommended as it may break the model or reduce TorchScript inference speed."
        )
        script = tr.jit.freeze(
            script, preserved_attrs=script.get_preserved_attributes()
        )
    if optimize:
        log.warning(f"Optimizing may break the model.")
        script = tr.jit.optimize_for_inference(script)
    check_for_preserved_attributes(script, script.get_preserved_attributes())
    return script


def save_neutone_model(
    model: "WaveformToWaveformBase",
    root_dir: Path,
    dump_samples: bool = False,
    submission: bool = False,
    audio_sample_pairs: List[AudioSamplePair] = None,
    max_n_samples: int = MAX_N_AUDIO_SAMPLES,
    freeze: bool = False,
    optimize: bool = False,
    speed_benchmark: bool = True,
) -> None:
    """
    Save a Neutone model to disk as a Torchscript file. Additionally include metadata file and samples as needed.

    Args:
        model: Your Neutone model. Should derive from neutone_sdk.WaveformToWaveformBase.
        root_dir: Directory to dump the model and auxiliary files.
        dump_samples: If true, will additionally dump audio samples from the .nm file for listening.
        submission: If true, will run additional checks to ensure the model
                saved on disk behaves identically to the one loaded in memory.
        audio_sample_pairs: Can be used to override the default samples bundled in the
                SDK to be added to the model. Will affect what is displayed on the
                website once the model is submitted.
        max_n_samples: Can be used to override the maximum number of samples
                used (default of 3) for exceptional cases.
        freeze: If true, jit.freeze will be applied to the model.
        optimize: If true, jit.optimize_for_inference will be applied to the model.
        speed_benchmark: If true, will run a speed benchmark when submission is also true.
                Consider disabling for non-realtime models as it might take too long.

    Returns:
      Will create the following files:
      ```
        root_dir/
        root_dir/model.nm
        root_dir/metadata.json
        root_dir/samples/*
      ```
    """
    random.seed(0)
    tr.manual_seed(0)
    if not model.use_debug_mode:
        log.warning(
            f"Debug mode has already been disabled for the model, please always test your model with debug "
            f"mode enabled."
        )

    root_dir.mkdir(exist_ok=True, parents=True)

    # TODO(cm): remove local import (currently prevents circular import)
    from neutone_sdk import SampleQueueWrapper
    from neutone_sdk.benchmark import benchmark_latency_, benchmark_speed_

    sqw = SampleQueueWrapper(model)

    with tr.inference_mode():
        log.info("Converting model to torchscript...")
        script = model_to_torchscript(sqw, freeze=freeze, optimize=optimize)

        # We need to keep a copy because some models still don't implement reset
        # properly and when rendering the samples we might create unwanted state.
        #
        # We used to deepcopy but we found it breaks some models
        # script_copy = copy.deepcopy(script)
        buf = io.BytesIO()
        tr.jit.save(script, buf)
        buf.seek(0)
        script_copy = tr.jit.load(buf)

        log.info("Extracting metadata...")
        metadata = script.to_metadata()._asdict()
        with open(root_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        if audio_sample_pairs is None:
            log.info("Running model on default audio samples...")
            input_samples = get_default_audio_samples()
            audio_sample_pairs = []
            for input_sample in input_samples:
                rendered_sample = render_audio_sample(sqw, input_sample)
                audio_sample_pairs.append(
                    AudioSamplePair(input_sample, rendered_sample)
                )

        metadata["sample_sound_files"] = [
            pair.to_metadata_format() for pair in audio_sample_pairs[:max_n_samples]
        ]
        log.info("Validating metadata...")
        validate_metadata(metadata)
        extra_files = {"metadata.json": json.dumps(metadata, indent=4).encode("utf-8")}

        # Save the copied model with the extra files
        log.info(f"Saving model to {root_dir/'model.nm'}...")
        tr.jit.save(script_copy, root_dir / "model.nm", _extra_files=extra_files)

        if dump_samples:
            dump_samples_from_metadata(metadata, root_dir)

        log.info("Loading saved model and metadata...")
        loaded_model, loaded_metadata = load_neutone_model(root_dir / "model.nm")
        check_for_preserved_attributes(
            loaded_model, loaded_model.get_preserved_attributes()
        )
        log.info("Testing methods used by the VST...")
        loaded_model.set_daw_sample_rate_and_buffer_size(48000, 512)
        loaded_model.reset()
        loaded_model.is_resampling()

        log.info("Testing offline mode...")
        for input_sample in get_default_audio_samples():
            offline_sr = input_sample.sr
            offline_bs = 4096
            loaded_model.set_daw_sample_rate_and_buffer_size(offline_sr, offline_bs)
            offline_params = tr.rand((MAX_N_PARAMS, input_sample.audio.size(1)))
            offline_rendered_sample = loaded_model.forward_offline(input_sample.audio, offline_params)
            break  # Only test one sample to reduce export time

        if submission:  # Do extra checks
            log.info("Running submission checks...")
            log.info("Assert metadata was saved correctly...")
            assert loaded_metadata == metadata
            del loaded_metadata["sample_sound_files"]
            assert loaded_metadata == loaded_model.to_metadata()._asdict()

            log.info(
                "Assert loaded model output matches output of model before saving..."
            )
            input_samples = audio_sample_pairs[0].input
            tr.manual_seed(42)
            loaded_model_render = render_audio_sample(loaded_model, input_samples).audio
            tr.manual_seed(42)
            script_model_render = render_audio_sample(script_copy, input_samples).audio

            assert tr.allclose(script_model_render, loaded_model_render, atol=1e-6)

            log.info("Running benchmarks...")
            log.info(
                "Check out the README for additional information on how to run benchmarks with different parameters and (sample_rate, buffer_size) combinations."
            )
            log.info("Running default latency benchmark...")
            benchmark_latency_(
                str(root_dir / "model.nm"),
            )
            if speed_benchmark:
                log.info(
                    "Running speed benchmark... If this is taking too long consider disabling the speed_benchmark parameter."
                )
                benchmark_speed_(str(root_dir / "model.nm"))
            else:
                log.info(
                    "Skipping speed_benchmark because the speed_benchmark parameter is set to False"
                )

            log.info("Your model has been exported successfully!")
            log.info(
                "You can now test it using the plugin available at https://neutone.space"
            )
            log.info(
                """Additionally, the parameter helper text is not displayed
                    correctly when using the local load functionality"""
            )
            log.info(
                """If you are happy with how your model sounds and would
            like to contribute it to the default list of models, please
            consider submitting it to our GitHub. Upload only the resulting model.nm
            somewhere and open an issue on GitHub using the Request add model
            template available at the following link:"""
            )
            log.info(
                "https://github.com/QosmoInc/neutone_sdk/issues/new?assignees=bogdanteleaga%2C+christhetree&labels=enhancement&template=request-add-model.md&title=%5BMODEL%5D+%3CNAME%3E"
            )


def load_neutone_model(path: str) -> Tuple[ScriptModule, Dict]:
    extra_files = {
        "metadata.json": "",
    }
    model = tr.jit.load(path, _extra_files=extra_files)
    loaded_metadata = json.loads(extra_files["metadata.json"].decode())
    assert validate_metadata(loaded_metadata)
    return model, loaded_metadata


def get_example_inputs(multichannel: bool = False) -> List[Tensor]:
    """
    returns a list of possible input tensors for an AuditionerModel.

    Possible inputs are audio tensors with shape (n_channels, n_samples).
    If multichannel == False, n_channels will always be 1.
    """
    max_channels = 2 if multichannel else 1
    num_inputs = 10
    channels = [random.randint(1, max_channels) for _ in range(num_inputs)]
    # sizes = [random.randint(2048, 396000) for _ in range(num_inputs)]
    sizes = [2048 for _ in range(num_inputs)]
    return [tr.rand((c, s)) for c, s in zip(channels, sizes)]


def test_run(model: "NeutoneModel", multichannel: bool = False) -> None:
    """
    Performs a couple of test forward passes with audio tensors of different sizes.
    Possible inputs are audio tensors with shape (n_channels, n_samples).
    If the model fails to meet the input/output requirements of either WaveformToWaveformBase or WaveformToLabelsBase,
      an assertion will be triggered by the respective class.

      Args:
        model (NeutoneModel): Your model, wrapped in either WaveformToWaveformBase or WaveformToLabelsBase
        multichannel (bool): if False, the number of input audio channels will always equal to 1. Otherwise,
                             some stereo test input arrays will be generated.
    Returns:

    """
    for x in get_example_inputs(multichannel):
        y = model(x)
        # plt.plot(y.cpu().numpy()[0])
        # plt.show()


def validate_waveform(x: Tensor, is_mono: bool) -> None:
    assert x.ndim == 2, "Audio tensor must have two dimensions: (channels, samples)"
    if is_mono:
        assert (
            x.shape[0] == 1
        ), "Audio tensor should be mono and have exactly one channel"
    else:
        assert (
            x.shape[0] == 2
        ), "Audio tensor should be stereo and have exactly two channels"
    assert x.shape[-1] > x.shape[0], (
        f"The number of channels {x.shape[-2]} exceeds the number of samples "
        f"{x.shape[-1]} in the audio tensor. There might be something "
        f"wrong with the model."
    )


def check_for_preserved_attributes(
    script: ScriptModule, preserved_attrs: List[str]
) -> None:
    for attr in preserved_attrs:
        assert hasattr(script, attr), (
            f"{attr}() method is missing from the TorchScript model. Did you overwrite "
            f"it and forget to add the @torch.jit.export decorator?"
        )
