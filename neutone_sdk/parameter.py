import logging
import os
from abc import ABC
from enum import Enum
from typing import Dict, List, Union, Optional

from neutone_sdk import constants

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class NeutoneParameterType(Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    TEXT = "text"


class NeutoneParameter(ABC):
    """
    Defines a Neutone Parameter abstract base class.

    The name and the description of the parameter will be shown as a tooltip
    within the UI. This parameter has no functionality and is meant to subclassed.
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
        """Returns a string dictionary containing the metadata of the parameter."""
        return {
            "name": self.name,
            "description": self.description,
            "default_value": str(self.default_value),
            "used": str(self.used),
            "type": self.type.value,
        }


class ContinuousNeutoneParameter(NeutoneParameter):
    """
    Defines a continuous Neutone Parameter that the user can use to control a model.

    The name and the description of the parameter will be shown as a tooltip
    within the UI.
    `default_value` must be between 0 and 1 and will be used as a default in the plugin
    when no presets are available.
    """

    def __init__(
        self, name: str, description: str, default_value: float, used: bool = True
    ):
        super().__init__(
            name,
            description,
            default_value,
            used,
            NeutoneParameterType.CONTINUOUS,
        )
        assert (
            0.0 <= default_value <= 1.0
        ), "`default_value` for continuous params must be between 0 and 1"


class CategoricalNeutoneParameter(NeutoneParameter):
    """
    Defines a categorical Neutone Parameter that the user can use to control a model.

    The name and the description of the parameter will be shown as a tooltip
    within the UI.
    `n_values` must be an int greater than or equal to 2 and less than or equal to
    `constants.MAX_N_CATEGORICAL_VALUES`.
    `default_value` must be in the range [0, `n_values` - 1].
    `labels` is a list of strings that will be used as the labels for the parameter.
    """

    def __init__(
        self,
        name: str,
        description: str,
        n_values: int,
        default_value: int,
        labels: Optional[List[str]] = None,
        used: bool = True,
    ):
        super().__init__(
            name, description, default_value, used, NeutoneParameterType.CATEGORICAL
        )
        assert 2 <= n_values <= constants.MAX_N_CATEGORICAL_VALUES, (
            f"`n_values` for categorical params must between 2 and "
            f"{constants.MAX_N_CATEGORICAL_VALUES}"
        )
        assert (
            0 <= default_value <= n_values - 1
        ), "`default_value` for categorical params must be between 0 and `n_values`-1"
        self.n_values = n_values
        if labels is None:
            labels = [str(idx) for idx in range(n_values)]
        else:
            assert len(labels) == self.n_values, "labels must have `n_values` elements"
        assert all(
            len(label) < constants.MAX_N_CATEGORICAL_LABEL_CHARS for label in labels
        ), (
            f"All labels must have length less than "
            f"{constants.MAX_N_CATEGORICAL_LABEL_CHARS} characters"
        )
        self.labels = labels

    def to_metadata_dict(self) -> Dict[str, str]:
        """Returns a string dictionary containing the metadata of the parameter."""
        data = super().to_metadata_dict()
        data["n_values"] = str(self.n_values)
        data["labels"] = "\t".join(self.labels)
        return data


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
