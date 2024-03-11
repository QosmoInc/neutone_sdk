# Neutone SDK

The Neutone SDK is a tool for researchers that enables them to wrap their own audio models and run them in a DAW using our [Neutone Plugin](https://neutone.space/). We offer functionality for both loading the models locally in the plugin and contributing them to the default list of models that is available to anyone running the plugin. We hope this will enable researchers to easily try their models in a DAW, but also provide creators with a collection of interesting models.

## Why use the Neutone SDK

[JUCE](https://github.com/juce-framework/JUCE) is the industry standard for building audio plugins. Because of this, knowledge of C++ is needed to be able to build even very simple audio plugins. However, it is rare for AI audio researchers to have extensive experience with C++ and be able to build such a plugin. Moreover, it is a serious time investment that could be spent developing better algorithms. Neutone makes it possible to build models using familiar tools such as PyTorch and with a minimal amount of Python code wrap these models such that they can be executed by the Neutone Plugin. Getting a model up and running inside a DAW can be done in less than a day without any need for C++ code or knowledge.

The SDK provides support for automatic buffering of inputs and outputs to your model and on-the-fly sample rate and stereo-mono conversion. It enables a model that can only be executed with a predefined number of samples to be used in the DAW at any sampling rate and any buffer size seamlessly. Additionally, within the SDK tools for benchmarking and profiling are readily available so you can easily debug and test the performance of your models.

<a name="examples"/>


## Table of Contents
- [Installing the SDK](#install)
- [Downloading the Neutone Plugin](#download)
- [Examples](#examples)
- [SDK Description](#description)
- [SDK Usage](#usage)
- [Benchmarking and Profiling](#benchmark)
- [Known issues](#issues)
- [Contributing to the SDK](#contributing)
- [Credits](#credits)

--- 

## Installing the SDK

<a name="install"/>

You can install `neutone_sdk` using pip: 

```
pip install neutone_sdk
```

<a name="download"/>

## Downloading the Plugin

The Neutone Plugin is available at [https://neutone.space](https://neutone.space). We currently offer VST3 and AU plugins that can be used to load the models created with this SDK. Please visit the website for more information.


## Examples and Notebooks

If you just want to wrap a model without going through a detailed description of what everything does we prepared these examples for you.

- The clipper example shows how to wrap a very simple PyTorch module that does not contain any AI model. Check it out for getting a high level overview of what is needed for wrapping a model. It is available at [examples/example_clipper.py](examples/example_clipper.py).
- An example with a simple convolutional model based on [Randomized Overdrive Neural Networks](https://csteinmetz1.github.io/ronn/) can be found at [examples/example_overdrive-random.py](examples/example_overdrive-random.py).
- We also have Notebooks for more complicated models showing the entire workflow from training to exporting them using Neutone:
    - [TCN FX Emulation](https://colab.research.google.com/drive/1gHZ-AEoYmfmWrjlKpKkK_SW1xzfxD24-?usp=sharing)
    - [DDSP Timbre Transfer](https://colab.research.google.com/drive/1yPHU6PRWw1lRWZLUxXimIa6chFQ2JdRW?usp=sharing)
    - [RAVE Timbre Transfer](https://colab.research.google.com/drive/1qlN6xLvDYrLcAwS8yh2ecmNG_bEKlVI9?usp=sharing)
    - [NoiseBandNet Audio Reconstruction](https://colab.research.google.com/drive/1KJij2CqhLf7ac6aljMckFL71WJrCNg66?usp=sharing)
    - [GCN FX Emulation](https://github.com/francescopapaleo/neural-audio-spring-reverb/blob/main/notebooks/neutone_GCN_demo.ipynb)


<a name="description"/>

## SDK Overview

The SDK provides functionality for wrapping existing PyTorch models in a way that can make them executable within the VST plugin. At its core the plugin is sending chunks of audio samples at a certain sample rate as an input and expects the same amount of samples at the output. The user of the SDK can specify what sample rate(s) and buffer size(s) their models perform optimally at. The SDK then guarantees that the forward pass of the model will receive audio at one of these (sample_rate, buffer_size) combinations. Four knobs are available that allow the users of the plugin to feed in additional parameters to the model at runtime. They can be enabled or disabled as needed via the SDK.


Using the included export function a series of tests is automatically ran to ensure the models behave as expected and are ready to be loaded by the plugin.


Benchmarking and profiling CLI tools are available for further debugging and testing of wrapped models. It is possible to benchmark the speed and latency of a model on a range of simulated common DAW (sample_rate, buffere_size) combinations as well as profile the memory and CPU usage.

<a name="usage"/>

## SDK Usage

### General Usage

We provide several models in the [examples](https://github.com/QosmoInc/neutone-sdk/blob/main/examples) directory. We will go through one of the simplest models, a distortion model, to illustrate.

Assume we have the following PyTorch model. Parameters will be covered later on, we will focus on the inputs and outputs for now. Assume this model receives a Tensor of shape `(2, buffer_size)` as an input where `buffer_size` is a parameter that can be specified.

```python
class ClipperModel(nn.Module):
    def forward(self, x: Tensor, min_val: float, max_val: float, gain: float) -> Tensor:
        return torch.clip(x, min=min_val * gain, max=max_val * gain)
```

To run this inside the VST the simplest wrapper we can write is by subclassing the WaveformToWaveformBase baseclass.
```python
class ClipperModelWrapper(WaveformToWaveformBase):
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
        # ... Parameter unwrap logic
        x = self.model.forward(x, min_val, max_val, gain)
        return x
 ```

The method that does most of the work is `do_forward_pass`. In this case it is just a simple passthrough, but we will use it to handle parameters later on.

By default the VST runs as `stereo-stereo` but when mono is desired for the model we can use the `is_input_mono` and `is_output_mono` to inform the SDK and have the inputs and outputs converted automatically. If `is_input_mono` is toggled an averaged `(1, buffer_size)` shaped Tensor will be passed as an input instead of `(2, buffer_size)`. If `is_output_mono` is toggled, `do_forward_pass` is expected to return a mono Tensor (shape `(1, buffer_size)`) that will then be duplicated across both channels at the output of the VST. This is done within the SDK to avoid unnecessary memory allocations during each pass.

`get_native_sample_rates` and `get_native_buffer_sizes` can be used to specify any preferred sample rates or buffer sizes. In most cases these are expected to only have one element but extra flexibility is provided for more complex models. In case multiple options are provided the SDK will try to find the best one for the current setting of the DAW. Whenever the sample rate or buffer size is different from the one of the DAW a wrapper is automatically triggered that converts to the correct sampling rate or implements a FIFO queue for the requested buffer size or both. This will incur a small performance penalty and add some amount of delay. In case a model is compatible with any sample rate and/or buffer size these lists can be left empty.

This means that the tensor `x` in the `do_forward_pass` method is guaranteed to be of shape `(1 if is_input_mono else 2, buffer_size)`  where `buffer_size` will be chosen at runtime from the list provided in the `get_native_buffer_sizes` method. The tensor `x` will also be at one of the sampling rates from the list provided in the `get_native_sample_rates` method.

### Exporting models and loading in the plugin

We provide a `save_neutone_model` helper function that saves models to disk. By default this will convert models to TorchScript and run them through a series of checks to ensure they can be loaded by the plugin. The resulting `model.nm` file can be loaded within the plugin using the `load your own` button. Read below for how to submit models to the default collection visible to everyone using the plugin.

### Parameters

For models that can use conditioning signals we currently provide four configurable knob parameters. Within the `ClipperModelWrapper` defined above we can include the following:
```python
class ClipperModelWrapper(WaveformToWaveformBase):
    ...
    
    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [NeutoneParameter(name="min", description="min clip threshold", default_value=0.5),
                NeutoneParameter(name="max", description="max clip threshold", default_value=1.0),
                NeutoneParameter(name="gain", description="scale clip threshold", default_value=1.0)]
         
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        min_val, max_val, gain = params["min"], params["max"], params["gain"]
        x = self.model.forward(x, min_val, max_val, gain)
        return x
```

During the forward pass the `params` variable will be a dictionary like the following:
```python
{
    "min": torch.Tensor([0.5] * buffer_size),
    "max": torch.Tensor([1.0] * buffer_size),
    "gain": torch.Tensor([1.0] * buffer_size)
}
```
The keys of the dictionary are specified in the `get_parameters` function.

The parameters will always take values between 0 and 1 and the `do_forward_pass` function can be used to do any necessary rescaling before running the internal forward method of the model.

Moreover, the parameters sent by the plugin come in at a sample level granularity. By default, we take the mean of each buffer and return a single float (as a Tensor), but the `aggregate_param` method can be used to override the aggregation method. See the full clipper export file for an example of preserving this granularity.

<a name="delay"/>

### Reporting delay

Some audio models will delay the audio for a certain amount of samples. This depends on the architecture of each particular model. In order for the wet and dry signal that is going through the plugin to be aligned users are required to report how many samples of delay their model induces. The `calc_model_delay_samples` can be used to specify the number of samples of delay. RAVE models on average have one buffer of delay (2048 samples) which is communicated statically in the `calc_model_delay_samples` method and can be seen in the examples. Models implemented with overlap-add will have a delay equal to the number of samples used for crossfading as seen in the [Demucs model wrapper](https://neutone.space/blog/implementing-models-with-overlap-add-in-neutone/) or the [spectral filter example](examples/example_spectral_filter.py). 

Calculating the delay your model adds can be difficult, especially since there can be multiple different sources of delay that need to be combined (e.g. cossfading delay, filter delay, lookahead buffer delay, and / or neural networks trained on unaligned dry and wet audio). It's worth spending some extra time testing the model in your DAW to make sure the delay is being reported correctly.

### Lookbehind Buffers

Adding a lookbehind buffer to your model can be useful for models that require a certain amount of additional context to output useful results. A lookbehind buffer can be enabled easily by indicating how many samples of lookbehind you need in the `get_look_behind_samples` method. When this method returns a number greater than zero, the `do_forward_pass` method will always receive a tensor of shape `(in_n_ch, look_behind_samples + buffer_size)`, but must still return a tensor of shape `(out_n_ch, buffer_size)` of the latest samples.

We recommend avoiding using a look-behind buffer when possible since it makes your model less efficient and can result in wasted calculations during each forward pass. If using a purely convolutional model, try switching all the convolutions to cached convolutions instead.

### Filters

It is common for AI models to act in unexpected ways when presented with inputs outside of the ones present in their training distribution. We provide a series of common filters (low bass, high pass, band pass, band stop) in the [neutone_sdk/filters.py](neutone_sdk/filters.py) file. These can be used during the forward pass to restrict the domain of the inputs going into the model. Some of them can induce a small amount of delay, check out the [examples/example_clipper_prefilter.py](examples/example_clipper_prefilter.py) file for a simple example on how to set up a filter.

### Submitting models

The plugin contains a default list of models aimed at creators that want to make use of them during their creative process. We encourage users to submit their models once they are happy with the results they get so they can be used by the community at large. For submission we require some additional metadata that will be used to display information about the model aimed at both creators and other researchers. This will be displayed on both the [Neutone website](https://neutone.space) and inside the plugin.

Skipping the previous clipper model, here is a more realistic example based on a random TCN overdrive model inspired by [micro-tcn](https://github.com/csteinmetz1/micro-tcn).

```python
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
        return "Random distortion/overdrive effect through randomly initialized Temporal-1D-convolution layers. You'll get different types of distortion by re-initializing the weight or changing the activation function. Based on the idea proposed by Steinmetz et al."

    def get_tags(self) -> List[str]:
        return ["distortion", "overdrive"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2010.04237",
            "Code": "https://github.com/csteinmetz1/ronn"
        }

    def get_citation(self) -> str:
        return "Steinmetz, C. J., & Reiss, J. D. (2020). Randomized overdrive neural networks. arXiv preprint arXiv:2010.04237."
```

Check out the documentation of the methods inside [core.py](neutone_sdk/core.py), as well as the random overdrive model on the [website](https://neutone.space/models/) and in the plugin to understand where each field will be displayed.

To submit a model, please [open an issue on the GitHub repository](https://github.com/QosmoInc/neutone_sdk/issues/new?assignees=bogdanteleaga%2C+christhetree&labels=enhancement&template=request-add-model.md&title=%5BMODEL%5D+%3CNAME%3E). We currently need the following:
- A short description of what the model does and how it can contribute to the community
- A link to the `model.nm` file outputted by the `save_neutone_model` helper function

<a name="benchmark"/>

## Benchmarking and Profiling

The SDK provides three CLI tools that can be used to debug and test wrapped models.

### Benchmarking Speed

Example:
```
$ python -m neutone_sdk.benchmark benchmark-speed --model_file model.nm
INFO:__main__:Running benchmark for buffer sizes (128, 256, 512, 1024, 2048) and sample rates (48000,). Outliers will be removed from the calculation of mean and std and displayed separately if existing.
INFO:__main__:Sample rate:  48000 | Buffer size:    128 | duration:  0.014±0.002 | 1/RTF:  5.520 | Outliers: [0.008]
INFO:__main__:Sample rate:  48000 | Buffer size:    256 | duration:  0.028±0.003 | 1/RTF:  5.817 | Outliers: []
INFO:__main__:Sample rate:  48000 | Buffer size:    512 | duration:  0.053±0.003 | 1/RTF:  6.024 | Outliers: []
INFO:__main__:Sample rate:  48000 | Buffer size:   1024 | duration:  0.106±0.000 | 1/RTF:  6.056 | Outliers: []
INFO:__main__:Sample rate:  48000 | Buffer size:   2048 | duration:  0.212±0.000 | 1/RTF:  6.035 | Outliers: [0.213]
```

Running the speed benchmark will automatically run random inputs through the model at a sample rate of 48000 and buffer sizes of (128, 256, 512, 1024, 2048) and report the average time taken to execute inference for one buffer. From this the `1/RTF` is calculated which represents how much faster than realtime the model is. As this number gets higher, the model will use fewer resources within the DAW. It is necessary for this number to be bigger than 1 for the model to be able to be executed in realtime on the machine that the benchmark is ran on.

The sample rates and buffer sizes being tested, as well as the number of times the benchmark is internally repeated to calculate the averages and the number of threads used for the computation are available as parameters. Run `python -m neutone_sdk.benchmark benchmark-speed --help` for more information. When specifiying custom sample rates or buffer sizes each individual one needs to be passed to the CLI separately. For example: `--sample_rate 48000 --sample_rate 44100 --buffer_size 32 --buffer_size 64`.

While the speed benchmark should be fast as the models are generally speaking required to be realtime it is possible to get stuck if the model is too slow. Make sure you choose an appropiate number of sample rates and buffer sizes to test.

<a name="latency"/>

### Benchmarking Latency

Example:
```bash
$ python -m neutone_sdk.benchmark benchmark-latency model.nm                    
INFO:__main__:Native buffer sizes: [2048], Native sample rates: [48000]
INFO:__main__:Model exports/ravemodel/model.nm has the following delays for each sample rate / buffer size combination (lowest delay first):
INFO:__main__:Sample rate:  48000 | Buffer size:   2048 | Total delay:      0 | (Buffering delay:      0 | Model delay:      0)
INFO:__main__:Sample rate:  48000 | Buffer size:   1024 | Total delay:   1024 | (Buffering delay:   1024 | Model delay:      0)
INFO:__main__:Sample rate:  48000 | Buffer size:    512 | Total delay:   1536 | (Buffering delay:   1536 | Model delay:      0)
INFO:__main__:Sample rate:  48000 | Buffer size:    256 | Total delay:   1792 | (Buffering delay:   1792 | Model delay:      0)
INFO:__main__:Sample rate:  44100 | Buffer size:    128 | Total delay:   1920 | (Buffering delay:   1920 | Model delay:      0)
INFO:__main__:Sample rate:  48000 | Buffer size:    128 | Total delay:   1920 | (Buffering delay:   1920 | Model delay:      0)
INFO:__main__:Sample rate:  44100 | Buffer size:    256 | Total delay:   2048 | (Buffering delay:   2048 | Model delay:      0)
INFO:__main__:Sample rate:  44100 | Buffer size:    512 | Total delay:   2048 | (Buffering delay:   2048 | Model delay:      0)
INFO:__main__:Sample rate:  44100 | Buffer size:   1024 | Total delay:   2048 | (Buffering delay:   2048 | Model delay:      0)
INFO:__main__:Sample rate:  44100 | Buffer size:   2048 | Total delay:   2048 | (Buffering delay:   2048 | Model delay:      0)
```

Running the speed benchmark will automatically compute the latency of the model at combinations of `sample_rate=(44100, 48000)` and `buffer_size=(128, 256, 512, 1024, 2048)`. This gives a general overview of what will happen for common DAW settings. The total delay is split into buffering delay and model delay. The model delay is reported by the creator of the model in the model wrapper as explained [above](#delay). The buffering delay is automatically computed by the SDK taking into consideration the combination of `(sample_rate, buffer_size)` specified by the wrapper (the native ones) and the one specified by the DAW at runtime. Running the model at its native `(sample_rate, buffer_size)` combination(s) will incur minimum delay.

Similar to the speed benchmark above, the tested combinations of `(sample_rate, buffer_size)` can be specified from the CLI. Run `python -m neutone_sdk.benchmark benchmark-latency --help` for more info.

### Profiling
```bash
$ python -m neutone_sdk.benchmark profile --model_file exports/ravemodel/model.nm
INFO:__main__:Profiling model exports/ravemodel/model.nm at sample rate 48000 and buffer size 128
STAGE:2023-09-28 14:34:53 96328:4714960 ActivityProfilerController.cpp:311] Completed Stage: Warm Up
30it [00:00, 37.32it/s]
STAGE:2023-09-28 14:34:54 96328:4714960 ActivityProfilerController.cpp:317] Completed Stage: Collection
STAGE:2023-09-28 14:34:54 96328:4714960 ActivityProfilerController.cpp:321] Completed Stage: Post Processing
INFO:__main__:Displaying Total CPU Time
INFO:__main__:--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         forward        98.54%     799.982ms       102.06%     828.603ms      26.729ms           0 b    -918.17 Kb            31  
               aten::convolution         0.12%     963.000us         0.95%       7.739ms     175.886us     530.62 Kb    -143.50 Kb            44
...
...
Full output removed from GitHub.

```

The profiling tool will run the model at a sample rate of 48000 and a buffer size of 128 under the PyTorch profiler and output a series of insights, such as the Total CPU Time, Total CPU Memory Usage (per function) and Grouped CPU Memory Usage (per group of function calls). This can be used to identify bottlenecks in your model code (even within the model call within the `do_forward_pass` call).

Similar to benchmarking, it can be ran at different combinations of sample rates and buffer sizes as well as different numbers of threads. Run `python -m neutone_sdk.benchmark profile --help` for more info.


<a name="issues"/>

## Known issues

- Freezing models on save can cause instabilities and thus freezing is disabled by default. We recommend trying to save models both with and without freeze.
- Displaying some metadata information does not currently work with local model loading in the plugin.
- Lookahead buffers will be implemented at the SDK level in the near future but is currently possible with additional code. An example is available in [this file](neutone_sdk/realtime_stft.py).
- M1 acceleration is currently not supported.
- Wrapping more complicated models can result in obscure TorchScript errors.


<a name="contributing"/>

## Contributing to the SDK

We welcome any contributions to the SDK. Please add types wherever possible and use the `black` formatter for readability.

The current roadmap is:
- Adding a TCN library and an overhaul of the TCN-based example models
- Supporting non-realtime models
- Supporting models with multiple inputs and / or outputs
- Supporting more and different types of parameters
- Looking into alternatives for TorchScript

<a name="credits"/>

## Credits

The audacitorch project was a major inspiration for the development of the SDK. [Check it out here](
https://github.com/hugofloresgarcia/audacitorch)
