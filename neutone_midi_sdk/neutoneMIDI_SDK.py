from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch as tr

from neutone_midi_sdk import (ContinuousNeutoneParameter, NeutoneMIDIModel,
                              NeutoneParameterType)
from neutone_midi_sdk.tokenization import (TokenData, convert_midi_to_tokens,
                                           convert_tokens_to_midi)


class MidiToMidiBase(NeutoneMIDIModel):
    def __init__(self,
                 model: tr.nn.Module,
                 vocab: Dict[str, int],
                 tokenizer_type: str,
                 tokenizer_data: TokenData,
                 add_dimension: bool = True):
        super().__init__(model, vocab, tokenizer_type, tokenizer_data)
        self.add_dimension = add_dimension

        assert all(
            p.type == NeutoneParameterType.CONTINUOUS or p.type == NeutoneParameterType.TENSOR
            for p in self.get_neutone_parameters()
        ), (
            "Only continuous or tensor type parameters are supported in MidiToMidiBase models. "
        )

        # For compatibility with the current plugin, we fill in missing params
        # TODO(nic): remove once plugin metadata parsing is implemented
        for idx in range(self.n_neutone_parameters, self.MAX_N_NUMERICAL_PARAMS):
            unused_p = ContinuousNeutoneParameter(
                name="",
                description="",
                default_value=0.0,
                used=False,
            )
            self.neutone_parameters_metadata[f"p{idx+1}"] = unused_p.to_metadata_dict()
            self.neutone_parameter_names.append(unused_p.name)
            self.neutone_parameter_descriptions.append(unused_p.description)
            self.neutone_parameter_types.append(unused_p.type.value)
            self.neutone_parameter_used.append(unused_p.used)

    def prepare_for_inference(self) -> None:
        super().prepare_for_inference()

    @abstractmethod
    def do_forward_pass(self, tokenized_data: tr.Tensor, params: Dict[str, tr.Tensor]) -> tr.Tensor:
        """
        SDK users can overwrite this method to implement the logic of their models.
        The input is a tensor of data that has been tokenized according to the tokenization settings,
        i.e. REMI, TSD, etc.

        In addition to the forward pass of your model, you can incorporate additional logic, such as
        the control parameters.

        The model should return data in the same format it was input, i.e. REMI-in, REMI-out. This will then
        be de-tokenized in the top-level 'forward' method.
        """
        pass

    def forward(self, midi_data: tr.Tensor, params: Optional[Dict[str, tr.Tensor]] = None) -> tr.Tensor:
        
        if params is None:
            # This codepath should never be reached, as the plugin always sends parameters. 
            params = self.get_default_param_values()

        for n in self.neutone_parameter_names:
            if n not in params:
                raise ValueError(f"Parameter {n} not found in input parameters.")
            self.remapped_params[n] = params[n]

        for p in self.neutone_parameters_metadata.keys():
            if self.neutone_parameters_metadata[p]["type"] == NeutoneParameterType.TENSOR.value and \
              self.neutone_parameters_metadata[p]["tokenize"] == str(True):
                name = self.neutone_parameters_metadata[p]["name"]
                # TODO: change this to token_type=self.tokenizer_type once deprecating HVO_taps
                self.remapped_params[name] = convert_midi_to_tokens(midi_data=params[name],
                                                        token_type="HVO",
                                                        midi_to_token_vocab=self.midi_to_token_vocab,
                                                        tokenizer_data=self.tokenizer_data)

        tokenized_data = convert_midi_to_tokens(midi_data=midi_data,
                                                token_type=self.tokenizer_type,
                                                midi_to_token_vocab=self.midi_to_token_vocab,
                                                tokenizer_data=self.tokenizer_data)


        if self.add_dimension:
            tokenized_data = tr.unsqueeze(tokenized_data, dim=0)
        model_output = self.do_forward_pass(tokenized_data, self.remapped_params)
        if self.add_dimension:
            model_output = tr.squeeze(model_output, dim=0)

        output_midi_data = convert_tokens_to_midi(tokens=model_output,
                                                  token_type=self.tokenizer_type,
                                                  token_to_midi_vocab=self.token_to_midi_vocab,
                                                  tokenizer_data=self.tokenizer_data)

        return output_midi_data
    
    def _get_numerical_default_param_values(
        self,
    ) -> List[Tuple[str, Union[float, int]]]:
        """
        Returns a list of tuples containing the name and default value of each
        numerical (float or int) parameter.
        For MidiToMidi models, there are always self.MAX_N_NUMERICAL_PARAMS number of
        numerical default parameter values, no matter how many parameters have been
        defined. This is to prevent empty tensors in some of the internal piping
        and queues when the model has no parameters.
        This should not be overwritten by SDK users.
        """
        result = []
        for p in self.get_neutone_parameters():
            if p.type == NeutoneParameterType.CONTINUOUS:
                result.append((p.name, p.default_value))
        if len(result) < self.MAX_N_NUMERICAL_PARAMS:
            result.extend(
                [
                    (f"p{idx + 1}", 0.0)
                    for idx in range(len(result), self.MAX_N_NUMERICAL_PARAMS)
                ]
            )
        return result
    
    def _get_tensor_default_param_values(
        self,
    ) -> List[Tuple[str, Union[tr.Tensor]]]:
        """
        Returns a list of tuples containing the name and default value of each
        tensor parameter.
        This should not be overwritten by SDK users.
        """
        result = []
        for p in self.get_neutone_parameters():
            if p.type == NeutoneParameterType.TENSOR:
                result.append((p.name, p.default_value))
        return result

def generate_fake_token_data():
    token_strings: Dict[str, List[str]] = {"value": ["value"]}
    token_floats: Dict[str, List[float]] = {"value": [0.0]}
    token_ints: Dict[str, List[int]] = {"value": [0]}
    token_data: TokenData = TokenData(token_strings, token_floats, token_ints)
    return token_data
