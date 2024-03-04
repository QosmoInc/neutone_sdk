import logging
import os
import time
from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, List, Tuple, Union

import torch as tr
from torch import nn, Tensor

from neutone_sdk import constants
from neutone_sdk.parameter import NeutoneParameter

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class CoreMetadata(NamedTuple):
    model_name: str
    model_authors: List[str]
    model_short_description: str
    model_long_description: str
    technical_description: str
    technical_links: Dict[str, str]
    neutone_parameters: Dict[str, Dict[str, str]]
    wet_default_value: float
    dry_default_value: float
    input_gain_default_value: float
    output_gain_default_value: float
    tags: List[str]
    model_version: str
    sdk_version: str
    date_created: float
    pytorch_version: str
    citation: str
    is_experimental: bool


class NeutoneModel(ABC, nn.Module):
    # TorchScript typing does not support instance attributes, so we need to type them
    # as class attributes. This is required for supporting models with no parameters.
    # (https://github.com/pytorch/pytorch/issues/51041#issuecomment-767061194)
    neutone_parameters_metadata: Dict[str, Dict[str, str]]
    remapped_params: Dict[str, Tensor]
    neutone_parameter_names: List[str]
    # TODO(cm): remove from here once plugin metadata parsing is implemented
    neutone_parameter_descriptions: List[str]
    neutone_parameter_used: List[bool]
    neutone_parameter_types: List[str]

    def __init__(self, model: nn.Module, use_debug_mode: bool = True) -> None:
        """
        Creates an Neutone model, wrapping a child model (that does the real
        work).
        """
        super().__init__()
        self.MAX_N_PARAMS = self._get_max_n_params()
        self.SDK_VERSION = constants.SDK_VERSION
        self.CURRENT_TIME = time.time()
        self.use_debug_mode = use_debug_mode
        self.n_neutone_parameters = len(self.get_neutone_parameters())

        # Ensure the number of parameters is within the allowed limit
        assert self.n_neutone_parameters <= self.MAX_N_PARAMS, (
            f"Number of parameters ({self.n_neutone_parameters}) exceeds the maximum "
            f"allowed ({self.MAX_N_PARAMS})."
        )
        # Ensure parameter names are unique
        assert len(set([p.name for p in self.get_neutone_parameters()])) == len(
            self.get_neutone_parameters()
        )

        # Save parameter metadata
        self.neutone_parameters_metadata = {
            f"p{idx + 1}": p.to_metadata_dict()
            for idx, p in enumerate(self.get_neutone_parameters())
        }

        # Allocate default params buffer to prevent dynamic allocations later
        numerical_default_param_vals = self._get_numerical_default_param_values()
        default_param_values_t = tr.tensor([v for _, v in numerical_default_param_vals])
        assert default_param_values_t.size(0) <= self.MAX_N_PARAMS, (
            f"Number of default parameter values ({default_param_values_t.size(0)}) "
            f"exceeds the maximum allowed ({self.MAX_N_PARAMS})."
        )
        default_param_values = default_param_values_t.unsqueeze(-1)
        self.register_buffer("default_param_values", default_param_values)

        # Allocate remapped params dictionary to prevent dynamic allocations later
        self.remapped_params = {
            name: tr.tensor([val])
            for name, val in numerical_default_param_vals
        }

        # Save parameter information
        self.neutone_parameter_names = [p.name for p in self.get_neutone_parameters()]
        # TODO(cm): remove from here once plugin metadata parsing is implemented
        self.neutone_parameter_descriptions = [
            p.description for p in self.get_neutone_parameters()
        ]
        self.neutone_parameter_used = [p.used for p in self.get_neutone_parameters()]
        self.neutone_parameter_types = [
            p.type.value for p in self.get_neutone_parameters()
        ]

        # Save and prepare model
        model.eval()
        self.model = model

    @abstractmethod
    def _get_max_n_params(self) -> int:
        """
        Sets the maximum number of parameters that the model can have.
        This should not be overwritten by SDK users.
        """
        pass

    @abstractmethod
    def _get_numerical_default_param_values(
        self,
    ) -> List[Tuple[str, Union[float, int]]]:
        """
        Returns a list of tuples containing the name and default value of each
        numerical (float or int) parameter.
        This should not be overwritten by SDK users.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Used to set the model name. This will be displayed on both the
        website and the plugin.

        Maximum length of 30 characters.
        """
        pass

    @abstractmethod
    def get_model_authors(self) -> List[str]:
        """
        Used to set the model authors. This will be displayed on both the
        website and the plugin.

        Should reflect the name of the people that developed the wrapper
        of the model using the SDK. Can be different from the authors of
        the original model.

        Maximum of 5 authors.
        """
        pass

    @abstractmethod
    def get_model_short_description(self) -> str:
        """
        Used to set the model short description. This will be displayed on both
        the website and the plugin.

        This is meant to be seen by the audio creators and should give a summary
        of what the model does.

        Maximum of 150 characters.
        """
        pass

    @abstractmethod
    def get_model_long_description(self) -> str:
        """
        Used to set the model long description. This will be displayed only on
        the website.

        This is meant to be seen by the audio creators and should give an extensive
        description of what the model does. Could describe interesting uses of the
        model, good combinations of parameters, what types of audio has it been
        tested with etc.

        Maximum of 500 characters.
        """
        pass

    @abstractmethod
    def get_technical_description(self) -> str:
        """
        Used to set the model technical description. This will be displayed only on
        the website.

        This is meant to be seen by other researchers or people that want to develop
        similar models. It could present a summary of the internals of the model:
        what architecture it is based on, what kind of data it was trained with,
        on what kind of hardware.

        If the authors of the plugin are different from the authors of the model(s)
        included this section along with citation and technical links are places
        to provide appropiate credits.

        Maximum of 500 characters.
        """
        pass

    @abstractmethod
    def get_tags(self) -> List[str]:
        """
        Used to provide a list of tags. This will be displayed on the website and will
        be used later on for filtering of similar models.

        Maximum of 7 tags of 15 characters each.
        """
        pass

    @abstractmethod
    def get_model_version(self) -> str:
        """
        Used to set the model version. This will be displayed on both the website and the plugin.

        We suggest people use semantic versioning for their models, but in a lot of cases it can
        be overkill. For now we only support showing the latest version of the model.

        Please provide a string like "1", "1.0", "1.0.0", "0.1.0" etc.
        """
        pass

    @abstractmethod
    def is_experimental(self) -> bool:
        """
        Used to set the experimental flag. This will be displayed on both the website and the plugin.

        If this flag is set the models will have a special icon next to them signaling to the users of
        the plugin that this model is an experimental release.
        """
        pass

    def get_technical_links(self) -> Dict[str, str]:
        """
        Used to set the hechnical links. These will be displayed only on the website.

        Under the technical description field the following links can be displayed as buttons.
        This can be used to provide links to the implementation, to scientific paper, personal websites etc.

        While any key-value pair can be provided, we strongly encourage users to provide a dictionary
        with keys such as Paper, Code, Personal, GitHub, Blog, Twitter, Instagram etc.

        Maximum of 3 links.
        """
        return {}

    def get_citation(self) -> str:
        """
        Used to set the citation. This will be displayed only on the website.

        This field is specifically meant to display the citation for a scientific paper that the model
        is based on, if any. Will be displayed under the technical links. Can be left empty.

        Maximum of 150 characters.
        """
        return ""

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return []

    def prepare_for_inference(self) -> None:
        """Prepare a model for inference and to be converted to torchscript."""
        self.use_debug_mode = False
        self.model.eval()
        self.eval()

    @tr.jit.export
    def get_neutone_parameters_metadata(self) -> Dict[str, Dict[str, str]]:
        """
        Returns the metadata of the parameters as a string dictionary of string
        dictionaries.
        """
        return self.neutone_parameters_metadata

    @tr.jit.export
    def get_default_param_values(self) -> Tensor:
        """
        Returns the default parameter values as a tensor of shape
        (N_DEFAULT_PARAM_VALUES, 1).
        """
        return self.default_param_values

    @tr.jit.export
    def get_default_param_names(self) -> List[str]:
        # TODO(cm): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_names

    @tr.jit.export
    def get_default_param_descriptions(self) -> List[str]:
        # TODO(cm): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_descriptions

    @tr.jit.export
    def get_default_param_types(self) -> List[str]:
        # TODO(cm): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_types

    @tr.jit.export
    def get_default_param_used(self) -> List[bool]:
        # TODO(cm): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_used

    @tr.jit.export
    def get_wet_default_value(self) -> float:
        return 1.0

    @tr.jit.export
    def get_dry_default_value(self) -> float:
        return 0.0

    @tr.jit.export
    def get_input_gain_default_value(self) -> float:
        """[0.0, 1.0] here maps to [-30.0db, +30.0db]"""
        return 0.5

    @tr.jit.export
    def get_output_gain_default_value(self) -> float:
        """[0.0, 1.0] here maps to [-30.0db, +30.0db]"""
        return 0.5

    @tr.jit.export
    def get_core_preserved_attributes(self) -> List[str]:
        return [
            "model",  # nn.Module
            "get_neutone_parameters_metadata",
            "get_default_param_values",
            "get_default_param_names",
            "get_default_param_descriptions",
            "get_default_param_types",
            "get_default_param_used",
            "get_wet_default_value",
            "get_dry_default_value",
            "get_input_gain_default_value",
            "get_output_gain_default_value",
            "get_core_preserved_attributes",
            "to_core_metadata",
        ]

    @tr.jit.export
    def to_core_metadata(self) -> CoreMetadata:
        return CoreMetadata(
            model_name=self.get_model_name(),
            model_authors=self.get_model_authors(),
            model_short_description=self.get_model_short_description(),
            model_long_description=self.get_model_long_description(),
            neutone_parameters=self.get_neutone_parameters_metadata(),
            wet_default_value=self.get_wet_default_value(),
            dry_default_value=self.get_dry_default_value(),
            input_gain_default_value=self.get_input_gain_default_value(),
            output_gain_default_value=self.get_output_gain_default_value(),
            technical_description=self.get_technical_description(),
            technical_links=self.get_technical_links(),
            tags=self.get_tags(),
            model_version=self.get_model_version(),
            sdk_version=self.SDK_VERSION,
            pytorch_version=tr.__version__,
            date_created=self.CURRENT_TIME,
            citation=self.get_citation(),
            is_experimental=self.is_experimental(),
        )
