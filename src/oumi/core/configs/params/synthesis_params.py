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

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.types.conversation import Conversation, Message, Role

_SUPPORTED_DATASET_FILE_TYPES = {".jsonl", ".json", ".csv", ".parquet", ".tsv"}


@dataclass
class TextMessage:
    """Text-only message to make it usable in omegaconf."""

    role: Role
    content: str

    def to_message(self) -> Message:
        """Convert to a Message."""
        return Message(role=self.role, content=self.content)


@dataclass
class TextConversation:
    """Text-only conversation to make it usable in omegaconf."""

    messages: list[TextMessage]

    conversation_id: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_conversation(self) -> Conversation:
        """Convert to a Conversation."""
        return Conversation(
            messages=[message.to_message() for message in self.messages],
            conversation_id=self.conversation_id,
            metadata=self.metadata,
        )


@dataclass
class DatasetSource:
    """Dataset to be used in synthesis."""

    path: str
    """Path to the dataset source."""

    hf_split: Optional[str] = None
    """Split of the huggingface dataset to be used in synthesis."""

    hf_revision: Optional[str] = None
    """Revision of the huggingface dataset to be used in synthesis."""

    attribute_map: Optional[dict[str, str]] = None
    """Map of attributes to be used in synthesis.
    Will use the existing keys in the dataset if not specified."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.path:
            raise ValueError("DatasetSource.path cannot be empty.")

        file_path = Path(self.path)
        prefix = self.path.split(":")[0]
        if prefix == "hf":
            return
        if file_path.suffix.lower() not in _SUPPORTED_DATASET_FILE_TYPES:
            raise ValueError(
                f"Unsupported dataset file type: {self.path}\n"
                f"Supported file types: {_SUPPORTED_DATASET_FILE_TYPES}"
            )


class SegmentationStrategy(str, Enum):
    """Segmentation strategies."""

    TOKENS = "tokens"
    """Segment the document via tokens."""


@dataclass
class DocumentSegmentationParams:
    """Segmentation parameters to be used when segmenting the document."""

    id: str
    """ID to be used when referencing the document segment during synthesis."""

    segmentation_strategy: SegmentationStrategy = SegmentationStrategy.TOKENS
    """Type of segmentation to be used."""

    tokenizer: str = "openai-community/gpt2"
    """Tokenizer to be used for segmentation.

    Tokenizers can be specified by their HuggingFace Hub ID or by direct file path.
    If not specified, will use the GPT-2 tokenizer from the HuggingFace Hub."""

    segment_length: int = 2048
    """Length of each segment, dependent on the segmentation strategy."""

    segment_overlap: int = 0
    """Overlap between segments. Must be less than segment_length."""

    keep_original_text: bool = False
    """Whether to keep the original text of the document."""

    def __post_init__(self):
        """Verifies/populates params."""
        if self.segment_length <= 0:
            raise ValueError("Segment length must be positive.")
        if self.segment_overlap < 0:
            raise ValueError("Segment overlap must be non-negative.")
        if self.segment_overlap >= self.segment_length:
            raise ValueError("Segment overlap must be less than segment length.")
        if self.segmentation_strategy == SegmentationStrategy.TOKENS:
            if not self.tokenizer:
                raise ValueError(
                    "DocumentSegmentationParams.tokenizer cannot be empty when "
                    "segmentation_strategy is TOKENS."
                )


@dataclass
class DocumentSource:
    """Documents to be used in synthesis."""

    path: str
    """Path to the document source."""

    id: str
    """ID to be used when referencing the document during synthesis."""

    segmentation_params: Optional[DocumentSegmentationParams] = None
    """Segmentation parameters to be used when segmenting the document."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.path:
            raise ValueError("DocumentSource.path cannot be empty.")
        if not self.id:
            raise ValueError("DocumentSource.id cannot be empty.")


@dataclass
class ExampleSource:
    """In-line examples to be used in synthesis."""

    examples: list[dict[str, Any]]
    """Examples to be used in synthesis."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.examples:
            raise ValueError("ExampleSource.examples cannot be empty.")

        keys = self.examples[0].keys()
        for example in self.examples:
            if example.keys() != keys:
                raise ValueError("All examples must have the same keys.")


@dataclass
class SampledAttributeValue:
    """Value to be sampled for the attribute."""

    id: str
    """ID to be used when referencing the attribute value during synthesis."""

    name: str
    """Plaintext name of the attribute value.
    Referenced as {attribute_id}"""

    description: str
    """Description of the attribute value.
    Referenced as {attribute_id.description}"""

    sample_rate: Optional[float] = None
    """Sample rate for the attribute value. If not specified, will assume uniform
    sampling among possible values."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise ValueError("SampledAttributeValue.id cannot be empty.")
        if not self.name:
            raise ValueError("SampledAttributeValue.name cannot be empty.")
        if not self.description:
            raise ValueError("SampledAttributeValue.description cannot be empty.")
        if self.sample_rate is not None and (
            self.sample_rate < 0 or self.sample_rate > 1
        ):
            raise ValueError(
                "SampledAttributeValue.sample_rate must be between 0 and 1."
            )


@dataclass
class SampledAttribute:
    """Attributes to be sampled across the dataset."""

    id: str
    """ID to be used when referencing the attribute during synthesis."""

    name: str
    """Plaintext name of the attribute. Referenced as {id.parent}"""

    description: str
    """Description of the attribute. Referenced as {id.parent.description}"""

    possible_values: list[SampledAttributeValue]
    """Values to be sampled for the attribute."""

    def get_value_distribution(self) -> dict[str, float]:
        """Get the distribution of attribute values."""
        value_distribution = {}
        for value in self.possible_values:
            value_distribution[value.id] = value.sample_rate
        return value_distribution

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise ValueError("SampledAttribute.id cannot be empty.")
        if not self.name:
            raise ValueError("SampledAttribute.name cannot be empty.")
        if not self.description:
            raise ValueError("SampledAttribute.description cannot be empty.")
        if not self.possible_values:
            raise ValueError("SampledAttribute.possible_values cannot be empty.")

        value_ids = []
        sample_rates = []
        for value in self.possible_values:
            value_ids.append(value.id)
            sample_rates.append(value.sample_rate)

        value_ids_set = set(value_ids)
        if len(value_ids) != len(value_ids_set):
            raise ValueError("SampledAttribute.possible_values must have unique IDs.")

        # Normalize sample rates
        normalized_sample_rates = []
        undefined_sample_rate_count = 0
        defined_sample_rate = 0.0
        for sample_rate in sample_rates:
            if sample_rate is not None:
                defined_sample_rate += sample_rate
            else:
                undefined_sample_rate_count += 1

            if defined_sample_rate > 1.0:
                raise ValueError("SampledAttribute.possible_values must sum to 1.0.")

        # Assign remaining sample rate to undefined sample rates
        remaining_sample_rate = 1.0 - defined_sample_rate
        for sample_rate in sample_rates:
            if sample_rate is None:
                normalized_sample_rates.append(
                    remaining_sample_rate / undefined_sample_rate_count
                )
            else:
                normalized_sample_rates.append(sample_rate)

        # Update sample rates
        for i, sample_rate in enumerate(normalized_sample_rates):
            self.possible_values[i].sample_rate = sample_rate


@dataclass
class AttributeCombination:
    """Sampling rates for combinations of attributes."""

    combination: dict[str, str]
    """Combination of attribute values to be used."""

    sample_rate: float
    """Sample rate for the combination."""

    def __post_init__(self):
        """Verifies/populates params."""
        if self.sample_rate < 0 or self.sample_rate > 1:
            raise ValueError(
                "AttributeCombination.sample_rate must be between 0 and 1."
            )
        if not self.combination:
            raise ValueError("AttributeCombination.combination cannot be empty.")

        for key, value in self.combination.items():
            if not key:
                raise ValueError(
                    "AttributeCombination.combination key cannot be empty."
                )
            if not value:
                raise ValueError(
                    "AttributeCombination.combination value cannot be empty."
                )

        if len(self.combination.keys()) <= 1:
            raise ValueError(
                "AttributeCombination.combination must have at least two keys."
            )


@dataclass
class GeneratedAttributePostprocessingParams:
    """Postprocessing parameters for generated attributes."""

    id: str
    """ID to be used when referencing the postprocessing parameters during synthesis."""

    keep_original_text_attribute: bool = True
    """Whether to keep the original text of the generated attribute.
    If True, the original text will be returned as an attribute.
    If False, the original text will be discarded."""

    cut_prefix: Optional[str] = None
    """Cut off value before and including prefix."""

    cut_suffix: Optional[str] = None
    """Cut off value after and including suffix."""

    regex: Optional[str] = None
    """Regex to be used to pull out the value from the generated text."""

    strip_whitespace: bool = True
    """Whether to strip whitespace from the value."""

    added_prefix: Optional[str] = None
    """Prefix to be added to the value."""

    added_suffix: Optional[str] = None
    """Suffix to be added to the value."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise ValueError(
                "GeneratedAttributePostprocessingParams.id cannot be empty."
            )

        if self.regex:
            try:
                re.compile(self.regex)
            except Exception as e:
                raise ValueError(
                    f"Error compiling GeneratedAttributePostprocessingParams.regex: {e}"
                )


@dataclass
class GeneratedAttribute:
    """Attributes to be generated."""

    id: str
    """ID to be used when referencing the attribute during synthesis."""

    instruction_messages: list[TextMessage]
    """List of messages providing instructions for generating this attribute."""

    postprocessing_params: Optional[GeneratedAttributePostprocessingParams] = None
    """Postprocessing parameters for the generated attribute."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise ValueError("GeneratedAttribute.id cannot be empty.")
        if not self.instruction_messages:
            raise ValueError("GeneratedAttribute.instruction_messages cannot be empty.")
        if self.postprocessing_params:
            if self.id == self.postprocessing_params.id:
                raise ValueError(
                    "GeneratedAttribute.id and "
                    "GeneratedAttributePostprocessingParams.id "
                    "cannot be the same."
                )


class TransformationType(str, Enum):
    """Types of transformation strategies."""

    STRING = "string"
    LIST = "list"
    DICT = "dict"
    CHAT = "chat"


@dataclass
class TransformationStrategy:
    """Discriminated union for transformation strategies that works with OmegaConf."""

    type: TransformationType
    """The type of transformation strategy."""

    # For string transformations
    string_transform: Optional[str] = None
    """String transformation template (used when type=STRING)."""

    # For list transformations
    list_transform: Optional[list[str]] = None
    """List of transforms for each element (used when type=LIST)."""

    # For dict transformations
    dict_transform: Optional[dict[str, str]] = None
    """Mapping of dictionary keys to their transforms (used when type=DICT)."""

    # For chat transformations
    chat_transform: Optional[TextConversation] = None
    """Chat transform for chat messages (used when type=CHAT)."""

    def __post_init__(self):
        """Verifies/populates params based on the type."""
        if self.type == TransformationType.STRING:
            if self.string_transform is None or self.string_transform == "":
                raise ValueError("string_transform cannot be empty when type=STRING")
            # Clear other fields
            self.list_transform = None
            self.dict_transform = None
            self.chat_transform = None

        elif self.type == TransformationType.LIST:
            if not self.list_transform or len(self.list_transform) == 0:
                raise ValueError("list_transform cannot be empty when type=LIST")
            # Clear other fields
            self.string_transform = None
            self.dict_transform = None
            self.chat_transform = None

        elif self.type == TransformationType.DICT:
            if not self.dict_transform or len(self.dict_transform) == 0:
                raise ValueError("dict_transform cannot be empty when type=DICT")
            # Clear other fields
            self.string_transform = None
            self.list_transform = None
            self.chat_transform = None

        elif self.type == TransformationType.CHAT:
            if not self.chat_transform or len(self.chat_transform.messages) == 0:
                raise ValueError("chat_transform cannot be empty when type=CHAT")

            messages = self.chat_transform.messages
            for message in messages:
                content = message.content
                if not isinstance(content, str):
                    raise ValueError("chat_transform message content must be a string")
                if not content:
                    raise ValueError("chat_transform message content cannot be empty")

            # Clear other fields
            self.string_transform = None
            self.list_transform = None
            self.dict_transform = None


@dataclass
class TransformedAttribute:
    """Transformation of existing attributes."""

    id: str
    """ID to be used when referencing the transformed attribute during synthesis."""

    transformation_strategy: TransformationStrategy
    """Strategy to be used for the transformation."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise ValueError("TransformedAttribute.id cannot be empty.")

        if not isinstance(self.transformation_strategy, TransformationStrategy):
            raise ValueError(
                "TransformedAttribute.transformation_strategy must be a "
                f"TransformationStrategy, got {type(self.transformation_strategy)}"
            )

    def get_strategy(self) -> TransformationStrategy:
        """Get the strategy for the transformation."""
        return self.transformation_strategy


@dataclass
class GeneralSynthesisParams(BaseParams):
    """General synthesis parameters."""

    input_data: Optional[list[DatasetSource]] = None
    """Datasets whose rows and columns will be used in synthesis.

    Rows will be enumerated during sampling, and columns can be referenced as attributes
    when generating new attributes."""

    input_documents: Optional[list[DocumentSource]] = None
    """Documents to be used in synthesis.

    Documents will be enumerated during sampling, and both documents and document
    segments can be referenced as attributes when generating new attributes."""

    input_examples: Optional[list[ExampleSource]] = None
    """In-line examples to be used in synthesis.

    Examples will be enumerated during sampling, and attributes can be referenced as
    attributes when generating new attributes."""

    sampled_attributes: Optional[list[SampledAttribute]] = None
    """Attributes to be varied across the dataset.

    Attributes each have a set of possible values which will be randomly sampled
    according to their sample rate. If no sample rate is specified, a uniform
    distribution is used. Sample rates must sum to <= 1.0. Any attributes that do not
    have a sample rate will be given a uniform sample rate equal to whatever remains.

    For example, if there are 3 attributes with sample rates of 0.5, 0.3, and 0.2,
    the total sample rate is 1.0. The first attribute will be sampled 50% of the time,
    the second attribute will be sampled 30% of the time, and the third attribute will
    be sampled 20% of the time. If the last two attributes have no sample rate, they
    will be sampled 25% of the time each as (1.0 - 0.5) / 2 = 0.25."""

    combination_sampling: Optional[list[AttributeCombination]] = None
    """Sampling rates for combinations of attributes.

    Each combination is a dictionary of attribute IDs to their values. The sample rate
    is the probability of sampling this combination. The sample rate of all combinations
    must sum to <= 1.0."""

    generated_attributes: Optional[list[GeneratedAttribute]] = None
    """Attributes to be generated.

    Generated attributes are created by running a chat with the model. The chat is
    specified by a list of messages. The messages will be populated with attribute
    values specific to that data point. The output of the chat is the generated
    attribute.

    For example, if one of the previous attributes is "name", and you use the following
    instruction messages::

        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do you pronounce the name {name}?"}
        ]

    Then assuming your data point has a value of "Oumi" for the "name" attribute, the
    chat will be run with the following messages::

        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do you pronounce the name Oumi?"}
        ]

    The model's response to these messages will be the value of the "name" attribute
    for that data point."""

    transformed_attributes: Optional[list[TransformedAttribute]] = None
    """Transformation of existing attributes.

    Transformed attributes involve no model interaction and instead are for the
    convenience of transforming parts of your data into a new form.

    For example, if you have "prompt" and "response" attributes, you can create a
    "chat" attribute by transforming the "prompt" and "response" attributes into a
    chat message::

        [
            {"role": "user", "content": "{prompt}"},
            {"role": "assistant", "content": "{response}"}
        ]

    """

    passthrough_attributes: Optional[list[str]] = None
    """When specified, will ONLY pass through these attributes in final output.
    If left unspecified, all attributes are saved. If an attribute is specified in
    passthrough_attributes but doesn't exist, it will be ignored."""

    def _check_attribute_ids(self, attribute_ids: set[str], id: str):
        """Check if the attribute ID is already in the set."""
        if id in attribute_ids:
            raise ValueError(
                f"GeneralSynthesisParams contains duplicate attribute IDs: {id}"
            )
        attribute_ids.add(id)

    def _check_dataset_source_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from dataset sources for uniqueness."""
        if self.input_data is None:
            return

        if len(self.input_data) == 0:
            raise ValueError("GeneralSynthesisParams.input_data cannot be empty.")

        for dataset_source in self.input_data:
            if dataset_source.attribute_map:
                for new_key in dataset_source.attribute_map.values():
                    self._check_attribute_ids(all_attribute_ids, new_key)

    def _check_document_source_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from document sources for uniqueness."""
        if self.input_documents is None:
            return

        if len(self.input_documents) == 0:
            raise ValueError("GeneralSynthesisParams.input_documents cannot be empty.")

        for document_source in self.input_documents:
            if not document_source.segmentation_params:
                continue

            seg_key = document_source.segmentation_params.id
            self._check_attribute_ids(all_attribute_ids, seg_key)

    def _check_example_source_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from example sources for uniqueness."""
        if self.input_examples is None:
            return

        if len(self.input_examples) == 0:
            raise ValueError("GeneralSynthesisParams.input_examples cannot be empty.")

        for example_source in self.input_examples:
            example_keys = example_source.examples[0].keys()
            for new_key in example_keys:
                self._check_attribute_ids(all_attribute_ids, new_key)

    def _check_sampled_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from sampled attributes for uniqueness."""
        if self.sampled_attributes is None:
            return

        if len(self.sampled_attributes) == 0:
            raise ValueError(
                "GeneralSynthesisParams.sampled_attributes cannot be empty."
            )

        for sampled_attribute in self.sampled_attributes:
            attribute_id = sampled_attribute.id
            self._check_attribute_ids(all_attribute_ids, attribute_id)

    def _check_generated_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from generated attributes for uniqueness."""
        if self.generated_attributes is None:
            return

        if len(self.generated_attributes) == 0:
            raise ValueError(
                "GeneralSynthesisParams.generated_attributes cannot be empty."
            )

        for generated_attribute in self.generated_attributes:
            attribute_id = generated_attribute.id
            self._check_attribute_ids(all_attribute_ids, attribute_id)
            if generated_attribute.postprocessing_params:
                postprocessing_id = generated_attribute.postprocessing_params.id
                self._check_attribute_ids(all_attribute_ids, postprocessing_id)

    def _check_transformed_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from transformed attributes for uniqueness."""
        if self.transformed_attributes is None:
            return

        if len(self.transformed_attributes) == 0:
            raise ValueError(
                "GeneralSynthesisParams.transformed_attributes cannot be empty."
            )

        for transformed_attribute in self.transformed_attributes:
            attribute_id = transformed_attribute.id
            self._check_attribute_ids(all_attribute_ids, attribute_id)

    def _check_combination_sampling_sample_rates(self) -> None:
        """Validate that the combination sample rates are <= 1.0."""
        if self.combination_sampling is None:
            return

        if len(self.combination_sampling) == 0:
            raise ValueError(
                "GeneralSynthesisParams.combination_sampling cannot be empty."
            )

        sample_rates = [
            combination.sample_rate for combination in self.combination_sampling
        ]
        if sum(sample_rates) > 1.0:
            raise ValueError(
                "GeneralSynthesisParams.combination_sampling sample rates must be "
                "less than or equal to 1.0."
            )

    def _check_passthrough_attribute_ids(self) -> None:
        """Validate that passthrough attributes are non-empty when defined."""
        if self.passthrough_attributes is None:
            return

        if len(self.passthrough_attributes) == 0:
            raise ValueError(
                "GeneralSynthesisParams.passthrough_attributes cannot be empty."
            )

    def __post_init__(self):
        """Verifies/populates params."""
        all_attribute_ids = set()
        self._check_dataset_source_attribute_ids(all_attribute_ids)
        self._check_document_source_attribute_ids(all_attribute_ids)
        self._check_example_source_attribute_ids(all_attribute_ids)
        self._check_sampled_attribute_ids(all_attribute_ids)
        self._check_generated_attribute_ids(all_attribute_ids)
        self._check_transformed_attribute_ids(all_attribute_ids)
        self._check_passthrough_attribute_ids()
        self._check_combination_sampling_sample_rates()
