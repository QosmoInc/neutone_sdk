import torch
from typing import Dict, List, Tuple


class TokenData:
    def __init__(self,
                 strings: Dict[str, List[str]],
                 floats: Dict[str, List[float]],
                 ints: Dict[str, List[int]]):
        self.strings = strings
        self.floats = floats
        self.ints = ints

    def get_elements(self) -> Tuple[Dict[str, List[str]], Dict[str, List[float]], Dict[str, List[int]]]:
        return self.strings, self.floats, self.ints


def convert_midi_to_tokens(midi_data: torch.Tensor,
                           token_type: str,
                           midi_to_token_vocab: Dict[str, int],
                           tokenizer_data: TokenData) \
        -> torch.Tensor:
    if token_type == "MIDILike":
        return convert_midi_to_midilike_tokens(midi_data, midi_to_token_vocab, tokenizer_data)

    elif token_type == "TSD":
        return convert_midi_to_tsd_tokens(midi_data, midi_to_token_vocab, tokenizer_data)

    elif token_type == "REMI":
        return convert_midi_to_remi_tokens(midi_data, midi_to_token_vocab, tokenizer_data)

    elif token_type == "HVO":
        return convert_midi_to_hvo(midi_data)
    
    elif token_type == "HVO_taps":
        return convert_midi_to_monophonic_hvo(midi_data)

    else:
        # Todo: Needs tensor return type; how to assert this?
        return torch.zeros((2, 2))


def convert_tokens_to_midi(tokens: torch.Tensor,
                           token_type: str,
                           token_to_midi_vocab: Dict[int, str],
                           tokenizer_data: TokenData) -> torch.Tensor:
    if token_type == "MIDILike":
        return convert_midilike_tokens_to_midi(tokens, token_to_midi_vocab)

    if token_type == "TSD":
        return convert_tsd_tokens_to_midi(tokens, token_to_midi_vocab)

    if token_type == "REMI":
        position_granularity: float = tokenizer_data.floats["pos_granularity"][0]
        return convert_remi_tokens_to_midi(tokens, token_to_midi_vocab, position_granularity)

    if token_type == "HVO":
        return convert_hvo_to_midi(tokens)

    else:
        return torch.zeros((2, 2))


"""
Utility Functions
Because torchscript scrictly enforces Typed python, it can often lead to very verbose code.
These functions perform common operations within the tokenisation methods while keeping things
fairly readable. 
"""


def closest_int(input_list: List[int], value: int) -> int:
    # https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    aux: List[int] = []
    for n in input_list:
        aux.append(abs(value - n))

    return input_list[aux.index(min(aux))]


def closest_float(input_list: List[float], value: float) -> float:
    aux: List[float] = []
    for n in input_list:
        aux.append(abs(value - n))

    return input_list[aux.index(min(aux))]


def closest_float_idx(input_list: List[float], value: float) -> int:
    aux: List[float] = []
    for n in input_list:
        aux.append(abs(value - n))

    return aux.index(min(aux))


def find_next_note_off_location(midi_data: torch.Tensor, note_value: int) -> float:
    location = 0.0
    for message in midi_data:
        if message[0] == 1.0 and int(message[1].item()) == note_value:
            location = float(message[3].item())
            return location
    return location


def extract_matching_strings(input_list: List[str], strings: List[str]) -> List[str]:
    output_list: List[str] = []
    for string in strings:
        for value in input_list:
            if string in value:
                output_list.append(value)
    return output_list


def calculate_delta(message: str) -> float:
    delta_string = message.split("_")[1]
    delta = int(delta_string.split(".")[0]) + int(delta_string.split(".")[1]) * (
            1 / int(delta_string.split(".")[2]))
    return delta


def convert_miditok_timing_to_float(message: str) -> float:
    timing_string = message.split("_")[1]
    float_value = float(timing_string.split(".")[0]) + float(timing_string.split(".")[1]) * (
            1 / float(timing_string.split(".")[2]))
    return float_value


"""
Individual tokenization methods are here. For each category of tokenization (i.e. CPWord, REMI, MIDILike, etc.)
there needs to be two functions: midi-to-token, and token-to-midi. 
We use an intermediary tuple representation of MIDI data:

(type(int), pitch(int), velocity(int), timestamp(float))
A note-on message with pitch of 64 and velocity of 90 on the 3rd beat of measure 5 would be:
(0, 64, 90, 5.75)

Therefor the tokenization method should convert this tuple to a string found within the vocab, and then
to the appropriate integer. Within the function, it should take care of any pitch/velocity/time quantization that is 
needed to fit within the given vocabulary. It will also likely be necessary to convert from the timeformat above to
that of the tokenization method. 
"""


# -------
# MIDILike
def convert_midi_to_midilike_tokens(midi_data: torch.Tensor,
                                    vocab: Dict[str, int],
                                    tokenizer_data: TokenData) -> torch.Tensor:
    """
    Given neunote MIDI data, a midi_to-token vocab, and available data*, convert the MIDI data
    into a tensor of tokens.
    *Available data (dict): [timeshift_strings, timeshift_floats, pitch_values, velocity_values]
    """

    token_strings: List[str] = []
    global_timestep: float = 0.0
    quantized_delta: float = 0.0

    for idx, message in enumerate(midi_data):

        pitch = int(message[1].item())
        pitch = closest_int(tokenizer_data.ints["pitch_values"], pitch)
        time = float(message[3].item())

        # Create timeshift token
        if time > global_timestep:
            delta = time - global_timestep
            if delta > (min(tokenizer_data.floats["timeshift_floats"]) * 0.5):
                closest_idx = closest_float_idx(tokenizer_data.floats["timeshift_floats"], delta)
                quantized_delta = tokenizer_data.floats["timeshift_floats"][closest_idx]
                token_strings.append(tokenizer_data.strings["timeshift_strings"][closest_idx])
                global_timestep += quantized_delta

        # Note on
        if message[0] == 0:
            velocity = int(message[2].item())
            velocity = closest_int(tokenizer_data.ints["velocity_values"], velocity)
            token_strings.append(f"NoteOn_{pitch}")
            token_strings.append(f"Velocity_{velocity}")

        # Note off
        elif message[0] == 1:
            token_strings.append(f"NoteOff_{pitch}")

    # Finally, convert the strings into ints per vocab dictionary
    tokens = [vocab[token] for token in token_strings]
    tokens = torch.tensor(tokens)

    return tokens


def convert_midilike_tokens_to_midi(tokens: torch.Tensor, tokens_to_midi_vocab: Dict[int, str]) -> torch.FloatTensor:
    midi_tuples_tensor_list: List[torch.FloatTensor] = []
    global_timestep = 0.0
    velocity = 90
    pitch = 64

    # Convert int tensor to intermediary string token representation
    string_tokens: List[str] = [tokens_to_midi_vocab[token.item()] for token in tokens]

    # Convert these messages into our final tuple format of (message_type, pitch, velocity, timestep)
    for idx, message in enumerate(string_tokens):

        if "NoteOn" in message and idx != len(string_tokens) - 1:
            if "Velocity" in string_tokens[idx + 1]:
                velocity = int(string_tokens[idx + 1].split("_")[1])
                pitch = int(message.split("_")[1])
                midi_tuples_tensor_list.append(torch.FloatTensor([[0.0,
                                                                   float(pitch),
                                                                   float(velocity),
                                                                   float(global_timestep)]]))

        elif "NoteOff" in message:
            pitch = int(message.split("_")[1])
            midi_tuples_tensor_list.append(torch.FloatTensor([[1.0,
                                                               float(pitch),
                                                               90.0,
                                                               float(global_timestep)]]))

        elif "TimeShift" in message:
            delta = convert_miditok_timing_to_float(message)
            global_timestep += delta

        else:
            pass

    midi_output_tensor: torch.FloatTensor = torch.cat(midi_tuples_tensor_list, 0)

    return midi_output_tensor


# -------
# TSD

def convert_midi_to_tsd_tokens(midi_data: torch.Tensor,
                               vocab: Dict[str, int],
                               tokenizer_data: TokenData) -> torch.Tensor:
    token_strings: List[str] = []
    global_timestep: float = 0.0

    for idx, message in enumerate(midi_data):

        pitch = int(message[1].item())
        pitch_quantized = closest_int(tokenizer_data.ints["pitch_values"], pitch)
        time = float(message[3].item())

        # Note on
        if message[0] == 0:

            # If later than current position, add timeshift token
            if time > global_timestep:
                delta = time - global_timestep
                if delta > (min(tokenizer_data.floats["timeshift_floats"]) * 0.5):
                    closest_idx = closest_float_idx(tokenizer_data.floats["timeshift_floats"], delta)
                    quantized_delta = tokenizer_data.floats["timeshift_floats"][closest_idx]
                    token_strings.append(tokenizer_data.strings["timeshift_strings"][closest_idx])
                    global_timestep += quantized_delta

            # Add pitch and velocity tokens
            velocity = int(message[2].item())
            velocity = closest_int(tokenizer_data.ints["velocity_values"], velocity)
            token_strings.append(f"Pitch_{pitch_quantized}")
            token_strings.append(f"Velocity_{velocity}")

            # Duration
            note_off_location: float = find_next_note_off_location(midi_data[idx:], note_value=pitch)
            delta = note_off_location - global_timestep
            closest_idx = closest_float_idx(tokenizer_data.floats["duration_floats"], delta)
            token_strings.append(tokenizer_data.strings["duration_strings"][closest_idx])

    tokens = [vocab[token] for token in token_strings]
    tokens = torch.tensor(tokens)
    return tokens


def convert_tsd_tokens_to_midi(tokens: torch.Tensor,
                               tokens_to_midi_vocab: Dict[int, str]) -> torch.FloatTensor:
    midi_tuples_tensor_list: List[torch.FloatTensor] = []
    global_timestep = 0.0
    velocity: float = 90.0
    pitch: float = 64.0

    # Convert int tensor to intermediary string token representation
    string_tokens: List[str] = [tokens_to_midi_vocab[token.item()] for token in tokens]

    for idx, message in enumerate(string_tokens):

        # For 'pitch' tokens, we need to ensure that the following two tokens
        # are velocity and duration (in either order). Otherwise the model
        # has not made a sequentially correct prediction and we will skip to the
        # next token
        if "Pitch" in message and idx != len(string_tokens) - 2:

            # Check if velocity and duration are present in the next 2 tokens, regardless of order
            if all(any(keyword in s for s in string_tokens[idx + 1:idx + 3]) for keyword in ("Velocity", "Duration")):
                matching_tokens = extract_matching_strings(string_tokens[idx + 1:idx + 3],
                                                           ["Velocity", "Duration"])
                vel_tok, dur_tok = matching_tokens[0], matching_tokens[1]

                pitch = float(message.split("_")[1])
                velocity = float(vel_tok.split("_")[1])
                duration = convert_miditok_timing_to_float(dur_tok)

                midi_tuples_tensor_list.append(torch.FloatTensor([[0.0,
                                                                   pitch,
                                                                   velocity,
                                                                   global_timestep]]))

                midi_tuples_tensor_list.append(torch.FloatTensor([[1.0,
                                                                   pitch,
                                                                   velocity,
                                                                   (global_timestep + duration)]]))

        elif "TimeShift" in message:
            global_timestep += calculate_delta(message)

    midi_output_tensor: torch.FloatTensor = torch.cat(midi_tuples_tensor_list, 0)

    return midi_output_tensor


# -------
# REMI


def convert_midi_to_remi_tokens(midi_data: torch.Tensor,
                                vocab: Dict[str, int],
                                tokenizer_data: TokenData) -> torch.Tensor:
    token_strings: List[str] = ["Bar_None"]
    global_timestep: float = 0.0
    new_bar: bool = False
    note_off_location: float = 0.0

    for idx, message in enumerate(midi_data):

        pitch = int(message[1].item())
        pitch_quantized = closest_int(tokenizer_data.ints["pitch_values"], pitch)
        time = float(message[3].item())

        if message[0] == 0:

            if time > global_timestep:
                delta = time - global_timestep

                # Todo: Remove hard-coded 4/4 timing. But Miditok only supports 4/4 as of June 2023
                # Deal with bar tokens
                while delta >= 4.0:
                    token_strings.append("Bar_None")
                    global_timestep += 4.0 - global_timestep % 4.0
                    delta = time - global_timestep
                    new_bar = True

                if delta > (min(tokenizer_data.floats["position_floats"]) * 0.5) or new_bar:
                    closest_idx = closest_float_idx(tokenizer_data.floats["position_floats"], delta)
                    quantized_delta = tokenizer_data.floats["position_floats"][closest_idx]
                    token_strings.append(tokenizer_data.strings["position_strings"][closest_idx])
                    global_timestep += quantized_delta

            velocity = int(message[2].item())
            velocity = closest_int(tokenizer_data.ints["velocity_values"], velocity)
            token_strings.append(f"Pitch_{pitch_quantized}")
            token_strings.append(f"Velocity_{velocity}")

            note_off_location = find_next_note_off_location(midi_data[idx:], note_value=pitch)
            delta = note_off_location - global_timestep
            closest_idx = closest_float_idx(tokenizer_data.floats["duration_floats"], delta)
            token_strings.append(tokenizer_data.strings["duration_strings"][closest_idx])

    tokens = [vocab[token] for token in token_strings]
    tokens = torch.tensor(tokens)

    return tokens


def convert_remi_tokens_to_midi(tokens: torch.Tensor,
                                tokens_to_midi_vocab: Dict[int, str],
                                position_granularity: float) -> torch.FloatTensor:
    midi_tuples_tensor_list: List[torch.FloatTensor] = []
    global_timestep = 0.0
    delta: float = 0.0
    velocity: float = 90.0
    pitch: float = 64.0

    # Convert int tensor to intermediary string token representation
    string_tokens: List[str] = [tokens_to_midi_vocab[token.item()] for token in tokens]

    # First "Bar_None" token can be removed.
    string_tokens = string_tokens[1:] if string_tokens[0] == "Bar_None" else string_tokens

    for idx, message in enumerate(string_tokens):

        if message == "Bar_None":
            global_timestep += 4.0 - global_timestep % 4.0

        if "Position" in message:
            global_timestep += float(message.split("_")[1]) * position_granularity

        if "Pitch" in message and idx != len(string_tokens) - 2:

            if all(any(keyword in s for s in string_tokens[idx + 1:idx + 3]) for keyword in ("Velocity", "Duration")):
                matching_tokens = extract_matching_strings(string_tokens[idx + 1:idx + 3],
                                                           ["Velocity", "Duration"])
                vel_tok, dur_tok = matching_tokens[0], matching_tokens[1]

                pitch = float(message.split("_")[1])
                velocity = float(vel_tok.split("_")[1])
                duration = convert_miditok_timing_to_float(dur_tok)

                midi_tuples_tensor_list.append(torch.FloatTensor([[0.0,
                                                                   pitch,
                                                                   velocity,
                                                                   global_timestep]]))

                midi_tuples_tensor_list.append(torch.FloatTensor([[1.0,
                                                                   pitch,
                                                                   velocity,
                                                                   (global_timestep + duration)]]))

    midi_output_tensor: torch.FloatTensor = torch.cat(midi_tuples_tensor_list, 0)

    return midi_output_tensor


def convert_midi_to_hvo(midi_data: torch.Tensor) -> torch.Tensor:
    # Determine total number of 2-bar patterns based on the highest time value in midi_data tensor
    mask = (midi_data[:, 0] == 0.0)
    num_patterns = int(torch.max(midi_data[mask, 3]) / 8) + 1
    hvo_tensor = torch.zeros((num_patterns, 32, 27))

    for idx, message in enumerate(midi_data):
        if float(message[0].item()) == 0:
            time = float(message[3].item())
            hit_location = int(round(time / 0.25) % 32)
            pattern = int(time / 8)
            velocity = float(message[2].item() / 127.0)

            # Check if the velocity is higher than previous input on this timestep
            # TODO: why is this indexed at 2, 11, 20?
            # TODO: is this checking the previous input on this timestep?
            if velocity > float(hvo_tensor[pattern, hit_location, 11].item()):
                offset = (time - (hit_location * 0.25)) / 0.125
                hvo_tensor[pattern, hit_location, 2] = 1.0
                hvo_tensor[pattern, hit_location, 11] = velocity
                hvo_tensor[pattern, hit_location, 20] = offset

    return hvo_tensor

def convert_midi_to_monophonic_hvo(midi_data: torch.Tensor) -> torch.Tensor:
    # Determine total number of 2-bar patterns as determined by the highest time value in midi_data tensor
    mask = (midi_data[:, 0] == 0.0)
    num_patterns = int(torch.max(midi_data[mask, 3]) / 8) + 1
    hvo_tensor = torch.zeros((num_patterns, 32, 3))

    for idx, message in enumerate(midi_data):
        if float(message[0].item()) == 0:
            time = float(message[3].item())
            hit_location = int(round(time / 0.25) % 32)
            pattern = int(time / 8)
            velocity = float(message[2].item() / 127.0)

            # Check if the velocity is higher than previous input on this timestep
            # TODO: Do we need a check here?
            # if velocity > float(hvo_tensor[pattern, hit_location - 1, 1].item()):
            offset = (time - (hit_location * 0.25)) / 0.125
            hvo_tensor[pattern, hit_location, 0] = 1.0
            hvo_tensor[pattern, hit_location, 1] = velocity
            hvo_tensor[pattern, hit_location, 2] = offset

    return hvo_tensor


def convert_hvo_to_midi(hvo: torch.Tensor) -> torch.Tensor:
    midi_tuples_tensor_list: List[torch.FloatTensor] = []
    roland_mapping = [36, 38, 42, 46, 43, 47, 50, 49, 51]

    # Input will be (x, 27, 32) where 'x' is the number of 2-bar patterns
    for pattern_idx, two_bar_sequence in enumerate(hvo):
        for beat_idx, step in enumerate(two_bar_sequence):
            for note_idx, note in enumerate(step[:9]):
                if note.item() >= 0.9:
                    pitch = float(roland_mapping[note_idx])
                    velocity = float(two_bar_sequence[beat_idx, (note_idx + 9)].item() * 127)
                    offset = float(two_bar_sequence[beat_idx, (note_idx + 18)].item() * 0.125)
                    time = float(beat_idx * 0.25) + offset
                    time = time if time >= 0.0 else 0.0
                    time += float(pattern_idx * 8.0)

                    # Note on
                    midi_tuples_tensor_list.append(torch.FloatTensor([[0.0,
                                                                       pitch,
                                                                       velocity,
                                                                       time]]))
                    # Note off
                    midi_tuples_tensor_list.append(torch.FloatTensor([[1.0,
                                                                       pitch,
                                                                       90.0,
                                                                       (time + 0.15)]]))
    if midi_tuples_tensor_list:
        midi_output_tensor: torch.Tensor = torch.cat(midi_tuples_tensor_list, 0)
        _, indices = torch.sort(midi_output_tensor[:, 3], descending=False)
        midi_output_tensor = midi_output_tensor[indices]
    else:
        midi_output_tensor: torch.Tensor = torch.zeros(2, 3)

    return midi_output_tensor
