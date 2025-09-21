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

"""Oumi (Open Universal Machine Intelligence) library.

This library provides tools and utilities for training, evaluating, and
inferring with machine learning models, particularly focused on language tasks.

Modules:
    - :mod:`~oumi.models`: Contains model architectures and related utilities.
    - :mod:`~oumi.evaluate`: Functions for evaluating models.
    - :mod:`~oumi.evaluate_async`: Asynchronous evaluation functionality.
    - :mod:`~oumi.infer`: Functions for model inference, including interactive mode.
    - :mod:`~oumi.train`: Training utilities for machine learning models.
    - :mod:`~oumi.utils`: Utility functions, including logging configuration.
    - :mod:`~oumi.judges`: Functions for judging datasets and conversations.

Functions:
    - :func:`~oumi.train.train`: Train a machine learning model.
    - :func:`~oumi.evaluate_async.evaluate_async`: Asynchronously evaluate a model.
    - :func:`~oumi.evaluate.evaluate`: Evaluate a model using LM Harness.
    - :func:`~oumi.infer.infer`: Perform inference with a trained model.
    - :func:`~oumi.infer.infer_interactive`: Run interactive inference with a model.
    - :func:`~oumi.quantize.quantize`: Quantize a model to reduce size and memory usage.
    - :func:`~oumi.judge.judge_dataset`: Judge a dataset using a model.

Examples:
    Training a model::

        >>> from oumi import train
        >>> from oumi.core.configs import TrainingConfig
        >>> config = TrainingConfig(...)
        >>> train(config)

    Evaluating a model::

        >>> from oumi import evaluate
        >>> from oumi.core.configs import EvaluationConfig
        >>> config = EvaluationConfig(...)
        >>> results = evaluate(config)

    Performing inference::

        >>> from oumi import infer
        >>> from oumi.core.configs import InferenceConfig
        >>> config = InferenceConfig(...)
        >>> outputs = infer(config)

    Quantizing a model::

        >>> from oumi import quantize
        >>> from oumi.core.configs import QuantizationConfig
        >>> config = QuantizationConfig(...)
        >>> result = quantize(config)

    Judging a dataset::
        >>> from oumi import judge_dataset
        >>> from oumi.core.configs import JudgeConfig
        >>> config = JudgeConfig(...)
        >>> judge_dataset(config, dataset)

See Also:
    - :mod:`oumi.core.configs`: For configuration classes used in Oumi
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from oumi.utils import logging

if TYPE_CHECKING:
    from oumi.core.configs import (
        AsyncEvaluationConfig,
        EvaluationConfig,
        InferenceConfig,
        JudgeConfig,
        QuantizationConfig,
        SynthesisConfig,
        TrainingConfig,
    )
    from oumi.core.inference import BaseInferenceEngine
    from oumi.core.types.conversation import Conversation
    from oumi.judges.base_judge import JudgeOutput
    from oumi.quantize.base import QuantizationResult

logging.configure_dependency_warnings()


def evaluate_async(config: AsyncEvaluationConfig) -> None:
    """Runs an async evaluation for a model using the provided configuration.

    Overview:
        This is a utility method for running evaluations iteratively over a series
        of checkpoints. This method can be run in parallel with a training job to
        compute metrics per checkpoint without wasting valuable time in the main
        training loop.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    import oumi.evaluate_async

    return oumi.evaluate_async.evaluate_async(config)


def evaluate(config: EvaluationConfig) -> list[dict[str, Any]]:
    """Evaluates a model using the provided configuration.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        A list of evaluation results (one for each task). Each evaluation result is a
        dictionary of metric names and their corresponding values.
    """
    import oumi.evaluate

    return oumi.evaluate.evaluate(config)


def infer_interactive(
    config: InferenceConfig,
    *,
    input_image_bytes: list[bytes] | None = None,
    system_prompt: str | None = None,
) -> None:
    """Interactively provide the model response for a user-provided input."""
    import oumi.infer

    return oumi.infer.infer_interactive(
        config, input_image_bytes=input_image_bytes, system_prompt=system_prompt
    )


def infer(
    config: InferenceConfig,
    inputs: list[str] | None = None,
    inference_engine: BaseInferenceEngine | None = None,
    *,
    input_image_bytes: list[bytes] | None = None,
) -> list[Conversation]:
    """Runs batch inference for a model using the provided configuration.

    Args:
        config: The configuration to use for inference.
        inputs: A list of inputs for inference.
        inference_engine: The engine to use for inference. If unspecified, the engine
            will be inferred from `config`.
        input_image_bytes: A list of input PNG image bytes to be used with `image+text`
            VLMs. Only used in interactive mode.

    Returns:
        object: A list of model responses.
    """
    import oumi.infer

    return oumi.infer.infer(
        config, inputs, inference_engine, input_image_bytes=input_image_bytes
    )


def judge_dataset(
    judge_config: JudgeConfig | str,
    dataset: list[dict[str, str]],
) -> list[JudgeOutput]:
    """Judge a dataset using Oumi's Judge framework.

    This function evaluates a dataset by instantiating a SimpleJudge with the provided
    configuration and running batch inference on all input data.

    The function performs the following steps:
        1. Initializes a SimpleJudge with the provided configuration.
        2. Passes the entire dataset to the judge for batch evaluation.
        3. Returns structured JudgeOutput objects containing parsed results.

    Args:
        judge_config: JudgeConfig object or path to a judge config file.
        dataset: List of dictionaries containing input data for evaluation. Each
            dictionary should contain key-value pairs that match placeholders in
            the judge's prompt template (e.g., {'question': '...', 'answer': '...'}).

    Returns:
        List[JudgeOutput]: A list of structured judgment results, each containing:
            - raw_output: The original response from the judge model
            - parsed_output: Extracted field values from structured formats (XML/JSON)
            - field_values: Typed values for each expected output field
            - field_scores: Numeric scores for applicable fields

    Example:
        >>> judge_config = JudgeConfig( # doctest: +SKIP
        ...     judge_params=JudgeParams(
        ...         prompt_template="Is this helpful? {question}, {answer}",
        ...         response_format=JudgeResponseFormat.XML,
        ...         judgment_type=JudgeOutputType.BOOL,
        ...         include_explanation=False
        ...     ),
        ...     inference_config=InferenceConfig(
        ...         model=ModelParams(model_name="gpt-4.1"),
        ...         generation=GenerationParams(max_tokens=100),
        ...         engine=InferenceEngineType.OPENAI
        ...     )
        ... )
        >>> dataset = [
        ...     {'question': 'What is 2+2?', 'answer': '4'},
        ...     {'question': 'How to cook?', 'answer': 'I dont know'}
        ... ]
        >>> judged_outputs = judge_dataset(judge_config, dataset)
        >>> for output in judged_outputs:
        ...     print(output.field_values)  # e.g., {'judgment': True}
    """
    import oumi.judge

    return oumi.judge.judge_dataset(judge_config, dataset)


def synthesize(config: SynthesisConfig) -> list[dict[str, Any]]:
    """Synthesize a dataset using the provided configuration."""
    import oumi.synth

    return oumi.synth.synthesize(config)


def train(
    config: TrainingConfig,
    additional_model_kwargs: dict[str, Any] | None = None,
    additional_trainer_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
) -> None:
    """Trains a model using the provided configuration."""
    import oumi.train

    return oumi.train.train(
        config,
        additional_model_kwargs=additional_model_kwargs,
        additional_trainer_kwargs=additional_trainer_kwargs,
        verbose=verbose,
    )


def quantize(config: QuantizationConfig) -> QuantizationResult:
    """Quantizes a model using the provided configuration.

    Args:
        config: Quantization configuration containing model parameters,
            method, output path, and other settings.

    Returns:
        QuantizationResult containing:
        - quantized_size_bytes: Size of the quantized model in bytes
        - output_path: Path to the quantized model
        - quantization_method: Quantization method used
        - format_type: Format type of the quantized model
        - additional_info: Additional method-specific information

    Raises:
        RuntimeError: If quantization fails for any reason
        ValueError: If configuration is invalid for this quantizer
    """
    import oumi.quantize

    return oumi.quantize.quantize(config)


__all__ = [
    "evaluate_async",
    "evaluate",
    "infer_interactive",
    "infer",
    "quantize",
    "synthesize",
    "train",
]
