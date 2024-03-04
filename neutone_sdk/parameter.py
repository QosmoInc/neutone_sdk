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
