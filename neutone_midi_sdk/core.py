import torch as tr
from torch import nn, Tensor
from typing import List, Dict, Tuple, Union
from abc import abstractmethod
from neutone_midi_sdk.tokenization import TokenData
from neutone_midi_sdk.parameter import NeutoneParameter
import neutone_midi_sdk.constants as constants


class NeutoneMIDIModel(tr.nn.Module):
    def __init__(self,
                 model: tr.nn.Module,
                 vocab: Dict[str, int],
                 tokenizer_type: str,
                 tokenizer_data: TokenData):

        super().__init__()
        self.MAX_N_NUMERICAL_PARAMS = constants.MAX_N_NUMERICAL_PARAMS
        self.MAX_N_TENSOR_PARAMS = constants.MAX_N_TENSOR_PARAMS
        self.SDK_VERSION = constants.SDK_VERSION
        self.n_neutone_parameters = len(self.get_neutone_parameters())

        # Allocate default numerical params to prevent dynamic allocations later
        numerical_default_param_vals = self._get_numerical_default_param_values()
        assert len(numerical_default_param_vals) <= self.MAX_N_NUMERICAL_PARAMS, (
            f"Number of default numerical parameter values ({len(numerical_default_param_vals)}) "
            f"exceeds the maximum allowed ({self.MAX_N_NUMERICAL_PARAMS})."
        )
        numerical_default_param_values_t = tr.tensor([v for _, v in numerical_default_param_vals])
        # Ensure number of parameters is within the maximum allowed
        self.n_numerical_neutone_parameters = len(numerical_default_param_vals)
        assert self.n_numerical_neutone_parameters <= self.MAX_N_NUMERICAL_PARAMS
        # Ensure parameter names are unique
        assert len(set([p.name for p in self.get_neutone_parameters()])) == len(
            self.get_neutone_parameters()
        )
        self.register_buffer("tensor_default_param_values", numerical_default_param_values_t.unsqueeze(-1))

        # Allocate default tensor params to prevent dynamic allocations later
        tensor_default_param_vals = self._get_tensor_default_param_values()
        assert len(tensor_default_param_vals) <= self.MAX_N_TENSOR_PARAMS, (
            f"Number of default tensor parameter values ({len(numerical_default_param_vals)}) "
            f"exceeds the maximum allowed ({self.MAX_N_TENSOR_PARAMS})."
        )
        # TODO(nic): this assumes a common dimension for all tensor parameters
        tensor_default_param_values_t = tr.cat([v for _, v in tensor_default_param_vals])
        self.register_buffer("numerical_default_param_values", tensor_default_param_values_t.unsqueeze(-1))

        # Save parameter metadata
        self.neutone_parameters_metadata = {
            f"p{idx + 1}": p.to_metadata_dict()
            for idx, p in enumerate(self.get_neutone_parameters())
        }

        # Allocate remapped params dictionary to prevent dynamic allocations later
        self.remapped_params = {
            name: tr.tensor([val])
            for name, val in numerical_default_param_vals
        }
        self.remapped_params.update(
            {
                name: val
                for name, val in tensor_default_param_vals
            }
        )
        self.default_param_values = self.remapped_params

        # Save parameter information
        self.neutone_parameter_names = [p.name for p in self.get_neutone_parameters()]
        # TODO(nic): remove from here once plugin metadata parsing is implemented
        self.neutone_parameter_descriptions = [
            p.description for p in self.get_neutone_parameters()
        ]
        self.neutone_parameter_used = [p.used for p in self.get_neutone_parameters()]
        self.neutone_parameter_types = [
            p.type.value for p in self.get_neutone_parameters()
        ]

        # instantiate model
        model.eval()
        self.model = model

        # Setup tokenization methods
        assert tokenizer_type in constants.SUPPORTED_TOKENIZATIONS, \
            f"{tokenizer_type} not a recognized tokenization format."
        tokenizer_data = generate_fake_token_data() if tokenizer_data is None else tokenizer_data
        vocab = {"v": 0} if vocab is None else vocab
        self.midi_to_token_vocab = vocab
        self.token_to_midi_vocab = {v: k for k, v in vocab.items()}
        self.tokenizer_type = tokenizer_type
        self.tokenizer_data: TokenData = TokenData(tokenizer_data.strings, tokenizer_data.floats, tokenizer_data.ints)

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
    def _get_tensor_default_param_values(
        self,
    ) -> List[Tuple[str, Union[float, int]]]:
        """
        Returns a list of tuples containing the name and default value of each
        tensor parameter.
        This should not be overwritten by SDK users.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Set the model name
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

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return []
    
    @tr.jit.export
    def get_neutone_parameters_metadata(self) -> Dict[str, Dict[str, str]]:
        """
        Returns the metadata of the parameters as a string dictionary of string
        dictionaries.
        """
        return self.neutone_parameters_metadata

    @tr.jit.export
    def get_default_param_values(self) -> Dict[str, Tensor]:
        """
        Returns the default parameter values as a tensor of shape
        (N_DEFAULT_PARAM_VALUES, 1).
        """
        return self.default_param_values
    
    @tr.jit.export
    def get_default_param_names(self) -> List[str]:
        # TODO(nic): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_names

    @tr.jit.export
    def get_default_param_descriptions(self) -> List[str]:
        # TODO(nic): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_descriptions

    @tr.jit.export
    def get_default_param_types(self) -> List[str]:
        # TODO(nic): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_types

    @tr.jit.export
    def get_default_param_used(self) -> List[bool]:
        # TODO(nic): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_used

    def prepare_for_inference(self) -> None:
        self.model.eval()
        self.eval()


# Todo: Would like to deprecate this method, it is used in "HVO" format where there is no TokenData necessary
def generate_fake_token_data():
    token_strings: Dict[str, List[str]] = {"value": ["value"]}
    token_floats: Dict[str, List[float]] = {"value": [0.0]}
    token_ints: Dict[str, List[int]] = {"value": [0]}
    token_data: TokenData = TokenData(token_strings, token_floats, token_ints)
    return token_data
