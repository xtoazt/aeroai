# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from oumi.core.configs.params.synthesis_params import GeneralSynthesisParams
from oumi.utils.placeholders import resolve_placeholders


class _AttributeParentInfo:
    """Information about a parent of a sampled attribute."""

    def __init__(self, parent_name: str, parent_description: str):
        """Initialize the attribute parent info."""
        self._parent_name = parent_name
        self.description = parent_description

    def __str__(self) -> str:
        return self._parent_name


class _AttributeInfo:
    """Information about a sampled attribute.

    Used to format the string for a sample.

    Example:
        attribute_id: "complexity"
        parent_name: "Complexity"
        parent_description: "The complexity of the text."
        value_name: "High"
        value_description: "The text is complex."

    Formatting string:
        {complexity.parent} ({complexity.parent.description})
        {complexity} ({complexity.description})

    Result:
        Complexity (The complexity of the text.)
        High (The text is complex.)
    """

    def __init__(
        self,
        attribute_id: str,
        parent_name: str,
        parent_description: str,
        value_name: str,
        value_description: str,
    ):
        """Initialize the attribute value info."""
        self.attribute_id = attribute_id
        self._attribute_name = value_name
        self.description = value_description
        self.parent = _AttributeParentInfo(parent_name, parent_description)

    def __str__(self) -> str:
        return self._attribute_name


class AttributeFormatter:
    """Formats a sample using a format string.

    Integrates information from permutable attributes to support
    formatting of placeholders in the format string (i.e. {attribute_id.value}).
    """

    def __init__(self, params: GeneralSynthesisParams):
        """Initialize the formatter."""
        self._params = params
        self._sampled_attribute_map = (
            {
                sampled_attr.id: sampled_attr
                for sampled_attr in params.sampled_attributes
            }
            if params.sampled_attributes
            else {}
        )
        self._sampled_attribute_info = {}

        # Pre-compute the attribute info for each possible value
        for attribute_id, attribute in self._sampled_attribute_map.items():
            for value in attribute.possible_values:
                key = (attribute_id, value.id)
                self._sampled_attribute_info[key] = _AttributeInfo(
                    attribute_id=attribute_id,
                    parent_name=attribute.name,
                    parent_description=attribute.description,
                    value_name=value.name,
                    value_description=value.description,
                )

    def format(
        self,
        sample: dict[str, str],
        format_string: str,
        missing_values_allowed: bool = False,
    ) -> str:
        """Format a sample using a format string.

        Args:
            sample: The sample to format.
            format_string: The format string to use.
            missing_values_allowed: If True, missing values are allowed in the sample.

        Returns:
            The formatted string.
        """
        attr_values = {}
        for attribute_id, attribute_value in sample.items():
            if self._is_sampled_attribute(attribute_id):
                value_id = attribute_value
                attr_values[attribute_id] = self._get_sampled_attribute_value_info(
                    attribute_id, value_id
                )
            else:
                attr_values[attribute_id] = attribute_value

        formatted_string = resolve_placeholders(
            format_string,
            attr_values,
            missing_values_allowed=missing_values_allowed,
        )
        return formatted_string

    def _is_sampled_attribute(self, attribute_id: str) -> bool:
        """Check if the attribute is a sampled attribute."""
        return attribute_id in self._sampled_attribute_map

    def _get_sampled_attribute_value_info(
        self, attribute_id: str, attribute_value_id: str
    ) -> _AttributeInfo:
        """Get the string representation information for a sampled attribute."""
        key = (attribute_id, attribute_value_id)
        if key in self._sampled_attribute_info:
            return self._sampled_attribute_info[key]

        raise ValueError(
            f"Attribute value {attribute_value_id} not found for "
            f"attribute {attribute_id}"
        )
