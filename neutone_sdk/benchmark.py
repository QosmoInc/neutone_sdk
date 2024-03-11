import os
import logging
import timeit
import itertools
from typing import List
import click
import torch
from torch.autograd.profiler import record_function
from neutone_sdk import constants
from neutone_sdk.sqw import SampleQueueWrapper
from neutone_sdk.utils import load_neutone_model, model_to_torchscript
import numpy as np
from tqdm import tqdm

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


@click.group()
def cli():
    """This is needed to make a group command with click."""
    pass


@cli.command()
@click.option("--model_file", help="Path to model file")
@click.option(
    "--buffer_size",
    default=(128, 256, 512, 1024, 2048),
    multiple=True,
    help="Buffer sizes to benchmark",
)
@click.option(
    "--sample_rate",
    default=(48000,),
    multiple=True,
    help="Sample rates to benchmark",
)
@click.option("--repeat", default=10, help="How many times to repeat the benchmark")
@click.option(
    "--n_iters",
    default=30,
    help="How many forward passes to run for each repetition",
)
@click.option(
    "--daw_is_mono",
    default=False,
    help="Whether to assume daw is mono or not during the benchmark",
)
@click.option("--num_threads", default=1, help="num_threads to use for the benchmark")
@click.option(
    "--num_interop_threads",
    default=1,
    help="num_interop_threads to use for the benchmark",
)
def benchmark_speed(
    model_file: str,
    buffer_size: List[int],
    sample_rate: List[int],
    repeat: int,
    n_iters: int,
    daw_is_mono: bool,
    num_threads: int,
    num_interop_threads: int,
) -> None:
    return benchmark_speed_(
        model_file,
        buffer_size,
        sample_rate,
        repeat,
        n_iters,
        daw_is_mono,
        num_threads,
        num_interop_threads,
    )


def benchmark_speed_(
    model_file: str,
    buffer_size: List[int] = (128, 256, 512, 1024, 2048),
    sample_rate: List[int] = (48000,),
    repeat: int = 10,
    n_iters: int = 30,
    daw_is_mono: bool = False,
    num_threads: int = 1,
    num_interop_threads: int = 1,
) -> None:
    daw_n_ch = 1 if daw_is_mono else 2
    np.set_printoptions(precision=3)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    with torch.no_grad():
        m, _ = load_neutone_model(model_file)
        log.info(
            f"Running benchmark for buffer sizes {buffer_size} and sample rates {sample_rate}. Outliers will be removed from the calculation of mean and std and displayed separately if existing."
        )
        for sr, bs in itertools.product(sample_rate, buffer_size):
            m.set_daw_sample_rate_and_buffer_size(sr, bs)
            for _ in range(n_iters):  # Warmup
                m.forward(torch.rand((daw_n_ch, bs)))
            m.reset()

            # Pregenerate random buffers to more accurately benchmark the model itself
            def get_random_buffer_generator():
                buffers = torch.rand(100, daw_n_ch, bs)
                i = 0

                def return_next_random_buffer():
                    nonlocal i
                    i = (i + 1) % 100
                    return buffers[i]

                return return_next_random_buffer

            rbg = get_random_buffer_generator()

            durations = np.array(
                timeit.repeat(lambda: m.forward(rbg()), repeat=repeat, number=n_iters)
            )
            m.reset()
            mean, std = np.mean(durations), np.std(durations)
            outlier_mask = np.abs(durations - mean) > 2 * std
            outliers = durations[outlier_mask]
            # Remove outliers from general benchmark
            durations = durations[~outlier_mask]
            mean, std = np.mean(durations), np.std(durations)
            log.info(
                f"Sample rate: {sr: 6} | Buffer size: {bs: 6} | duration: {mean: 6.3f}±{std:.3f} | 1/RTF: {bs/(mean/n_iters*sr): 6.3f} | Outliers: {outliers[:3]}"
            )


@cli.command()
@click.option("--model_file", help="Path to model file")
@click.option(
    "--buffer_size",
    default=(128, 256, 512, 1024, 2048),
    multiple=True,
    help="Buffer sizes to benchmark",
)
@click.option(
    "--sample_rate",
    default=(
        44100,
        48000,
    ),
    multiple=True,
    help="Sample rates to benchmark",
)
def benchmark_latency(
    model_file: str, buffer_size: List[int], sample_rate: List[int]
) -> None:
    return benchmark_latency_(model_file, buffer_size, sample_rate)


def benchmark_latency_(
    model_file: str,
    buffer_size: List[int] = (128, 256, 512, 1024, 2048),
    sample_rate: List[int] = (48000,),
) -> None:
    m, _ = load_neutone_model(model_file)
    nbs, nsr = m.get_native_buffer_sizes(), m.get_native_sample_rates()
    log.info(f"Native buffer sizes: {nbs[:10]}, Native sample rates: {nsr[:10]}")
    if len(nbs) > 10 or len(nsr) > 10:
        log.info(f"Showing only the first 10 values in case there are more.")
    with torch.no_grad():
        delays = []
        for sr, bs in itertools.product(sample_rate, buffer_size):
            m.set_daw_sample_rate_and_buffer_size(sr, bs)
            m.reset()
            delays += [
                [
                    sr,
                    bs,
                    m.calc_buffering_delay_samples(),
                    m.calc_model_delay_samples(),
                ]
            ]
        delays = sorted(delays, key=lambda x: x[2] + x[3])
        log.info(
            f"Model {model_file} has the following delays for each sample rate / buffer size combination (lowest delay first):"
        )
        for sr, bs, bds, mds in delays:
            log.info(
                f"Sample rate: {sr: 6} | Buffer size: {bs: 6} | Total delay: {bds+mds: 6} | (Buffering delay: {bds: 6} | Model delay: {mds: 6})"
            )
        log.info(
            f"The recommended sample rate / buffer size combination is sample rate {delays[0][0]}, buffer size {delays[0][1]}"
        )


def profile_sqw(
    sqw: SampleQueueWrapper,
    daw_sr: int = 48000,
    daw_bs: int = 512,
    daw_is_mono: bool = False,
    use_params: bool = True,
    convert_to_torchscript: bool = False,
    n_iters: int = 100,
) -> None:
    daw_n_ch = 1 if daw_is_mono else 2
    audio_buffers = [torch.rand((daw_n_ch, daw_bs)) for _ in range(n_iters)]
    if use_params:
        param_buffers = [
            torch.rand((constants.MAX_N_PARAMS, daw_bs)) for _ in range(n_iters)
        ]
    else:
        param_buffers = [None for _ in range(n_iters)]

    sqw.set_daw_sample_rate_and_buffer_size(daw_sr, daw_bs)
    if hasattr(sqw, "prepare_for_inference"):
        sqw.prepare_for_inference()
    if convert_to_torchscript:
        log.info("Converting to TorchScript")
        with torch.no_grad():
            sqw = model_to_torchscript(sqw, freeze=False, optimize=False)

    with torch.inference_mode():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_stack=True,
            profile_memory=True,
            record_shapes=False,
        ) as prof:
            with record_function("forward"):
                for audio_buff, param_buff in tqdm(zip(audio_buffers, param_buffers)):
                    out_buff = sqw.forward(audio_buff, param_buff)

        log.info("Displaying Total CPU Time")
        log.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # log.info(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
        log.info("Displaying CPU Memory Usage")
        log.info(
            prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10)
        )
        log.info("Displaying Grouped CPU Memory Usage")
        log.info(
            prof.key_averages(group_by_stack_n=5).table(
                sort_by="self_cpu_memory_usage", row_limit=5
            )
        )


@cli.command()
@click.option("--model_file", help="Path to model file")
@click.option(
    "--buffer_size",
    default=(128,),
    multiple=True,
    help="Buffer sizes to benchmark",
)
@click.option(
    "--sample_rate",
    default=(48000,),
    multiple=True,
    help="Sample rates to benchmark",
)
@click.option(
    "--daw_is_mono",
    default=False,
    help="Whether to assume daw is mono or not during the benchmark",
)
@click.option(
    "--use_params",
    default=False,
    help="Whether to pass parameters to the model during profiling",
)
@click.option(
    "--n_iters",
    default=30,
    help="How many forward passes to run while profiling",
)
@click.option("--num_threads", default=1, help="num_threads to use for the benchmark")
@click.option(
    "--num_interop_threads",
    default=1,
    help="num_interop_threads to use for the benchmark",
)
def profile(
    model_file: str,
    buffer_size: List[int],
    sample_rate: List[int],
    daw_is_mono: bool = False,
    use_params: bool = True,
    n_iters: int = 100,
    num_threads: int = 1,
    num_interop_threads: int = 1,
):
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    m, _ = load_neutone_model(model_file)
    for sr, bs in itertools.product(sample_rate, buffer_size):
        log.info(
            f"Profiling model {model_file} at sample rate {sr} and buffer size {bs}"
        )
        profile_sqw(
            m,
            sr,
            bs,
            daw_is_mono,
            use_params,
            False,
            n_iters,
        )


if __name__ == "__main__":
    cli()
