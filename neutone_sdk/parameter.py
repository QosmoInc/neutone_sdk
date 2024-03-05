import logging
import os
from abc import ABC
from enum import Enum
from typing import Dict, Union

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class NeutoneParameterType(Enum):
    KNOB = "knob"
    TEXT = "text"


class NeutoneParameter(ABC):
    """
    Defines a Neutone Parameter abstract base class.

    The name and the description of the parameter will be shown as a tooltip
    within the UI. This parameter has no functionality.
    """

    def __init__(
        self,
        name: str,
        description: str,
        default_value: Union[int, float, str],
        used: bool,
        param_type: NeutoneParameterType,
    ):
        self.name = name
        self.description = description
        self.default_value = default_value
        self.used = used
        self.type = param_type

    def to_metadata_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "default_value": str(self.default_value),
            "used": str(self.used),
            "type": self.type.value,
        }


class KnobNeutoneParameter(NeutoneParameter):
    """
    Defines a knob Neutone Parameter that the user can use to control a model.

    The name and the description of the parameter will be shown as a tooltip
    within the UI. `default_value` must be between 0 and 1 and will be used
    as a default in the plugin when no presets are available.
    """

    def __init__(
        self, name: str, description: str, default_value: float, used: bool = True
    ):
        super().__init__(
            name,
            description,
            default_value,
            used,
            NeutoneParameterType.KNOB,
        )


class TextNeutoneParameter(NeutoneParameter):
    """
    Defines a text Neutone Parameter that the user can use to control a model.

    The name and the description of the parameter will be shown as a tooltip
    within the UI.
    `max_n_chars` specifies the maximum number of characters that the user can input.
    If this value is set to -1, there is no limit on the number of characters.
    `default_value` is the default value to be automatically populated in the text box.
    """

    def __init__(
        self,
        name: str,
        description: str,
        max_n_chars: int = -1,
        default_value: str = "",
        used: bool = True,
    ):
        super().__init__(
            name, description, default_value, used, NeutoneParameterType.TEXT
        )
        assert max_n_chars >= -1, "`max_n_chars` must be greater than or equal to -1"
        if max_n_chars != -1:
            assert (
                len(default_value) <= max_n_chars
            ), "`default_value` must be a string of length less than `max_n_chars`"
        self.max_n_chars = max_n_chars

    def to_metadata_dict(self) -> Dict[str, str]:
        """Returns a string dictionary containing the metadata of the parameter."""
        data = super().to_metadata_dict()
        data["max_n_chars"] = str(self.max_n_chars)
        return data
