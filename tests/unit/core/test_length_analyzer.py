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

"""Tests for the LengthAnalyzer."""

from unittest.mock import Mock

import pytest

from oumi.core.analyze.length_analyzer import LengthAnalyzer
from oumi.core.types.conversation import Conversation, Message, Role


def _single_message_conversation(text):
    return Conversation(messages=[Message(role=Role.USER, content=text)])


def test_char_count():
    """Test character count functionality."""
    analyzer = LengthAnalyzer(
        char_count=True, word_count=False, sentence_count=False, token_count=False
    )
    conv = _single_message_conversation("Hello, world!")
    message_results, conversation_result = analyzer.analyze_sample(conv)
    assert message_results[0].analyzer_metrics["char_count"] == 13
    # Only char_count should be present
    assert len(message_results[0].analyzer_metrics) == 1


def test_word_count():
    """Test word count functionality."""
    analyzer = LengthAnalyzer(
        char_count=False, word_count=True, sentence_count=False, token_count=False
    )
    conv = _single_message_conversation("Hello world! This is a test.")
    message_results, conversation_result = analyzer.analyze_sample(conv)
    assert message_results[0].analyzer_metrics["word_count"] == 6
    # Only word_count should be present
    assert len(message_results[0].analyzer_metrics) == 1


def test_sentence_count():
    """Test sentence count functionality."""
    analyzer = LengthAnalyzer(
        char_count=False, word_count=False, sentence_count=True, token_count=False
    )
    conv = _single_message_conversation("Hello world! This is a test. How are you?")
    message_results, conversation_result = analyzer.analyze_sample(conv)
    assert message_results[0].analyzer_metrics["sentence_count"] == 3
    # Only sentence_count should be present
    assert len(message_results[0].analyzer_metrics) == 1


def test_analyzer_instantiation():
    """Test analyzer can be instantiated with different parameter combinations."""
    # Test with defaults
    analyzer = LengthAnalyzer()
    conv = _single_message_conversation("Hello, world!")
    message_results, conversation_result = analyzer.analyze_sample(conv)
    assert message_results[0].analyzer_metrics["char_count"] == 13
    assert message_results[0].analyzer_metrics["word_count"] == 2
    assert message_results[0].analyzer_metrics["sentence_count"] == 1
    assert "token_count" not in message_results[0].analyzer_metrics

    # Test with custom parameters
    analyzer = LengthAnalyzer(
        char_count=True, word_count=False, sentence_count=True, token_count=False
    )
    conv = _single_message_conversation("Hello, world!")
    message_results, conversation_result = analyzer.analyze_sample(conv)
    assert message_results[0].analyzer_metrics["char_count"] == 13
    assert "word_count" not in message_results[0].analyzer_metrics
    assert message_results[0].analyzer_metrics["sentence_count"] == 1
    assert "token_count" not in message_results[0].analyzer_metrics

    # Test with partial parameters (some defaults, some overridden)
    analyzer = LengthAnalyzer(char_count=False, word_count=True)
    conv = _single_message_conversation("Hello, world!")
    message_results, conversation_result = analyzer.analyze_sample(conv)
    assert "char_count" not in message_results[0].analyzer_metrics
    assert message_results[0].analyzer_metrics["word_count"] == 2
    assert message_results[0].analyzer_metrics["sentence_count"] == 1  # Default True
    assert "token_count" not in message_results[0].analyzer_metrics  # Default False


def test_token_count():
    """Test token count functionality."""
    # Test token count with tokenizer only (default: includes special tokens)
    mock_tokenizer = Mock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_tokenizer.encode.return_value = [0, 1, 2, 3, 4, 5, 2]  # 7 tokens

    analyzer = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer,
    )
    conv = _single_message_conversation("Hello, world!")
    message_results, conversation_result = analyzer.analyze_sample(conv)
    assert message_results[0].analyzer_metrics["token_count"] == 7
    # analyze_sample calls tokenizer once for message and once for conversation
    assert mock_tokenizer.encode.call_count == 2
    assert mock_tokenizer.apply_chat_template.call_count == 1
    # Check that it was called with the message text
    mock_tokenizer.encode.assert_any_call("Hello, world!", add_special_tokens=True)
    # Check that apply_chat_template was called with the conversation
    mock_tokenizer.apply_chat_template.assert_called_with(
        conv,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Test without special tokens (explicitly set to False)
    mock_tokenizer_no_special = Mock()
    mock_tokenizer_no_special.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

    analyzer_no_special = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer_no_special,
        include_special_tokens=False,
    )
    conv = _single_message_conversation("Hello, world!")
    message_results, conversation_result = analyzer_no_special.analyze_sample(conv)
    assert message_results[0].analyzer_metrics["token_count"] == 5
    # Check that it was called without special tokens
    mock_tokenizer_no_special.encode.assert_any_call(
        "Hello, world!", add_special_tokens=False
    )

    # Test without tokenizer (should raise ValueError)
    with pytest.raises(ValueError, match="tokenizer must be provided"):
        analyzer_no_tokenizer = LengthAnalyzer(
            char_count=False,
            word_count=False,
            sentence_count=False,
            token_count=True,
            # No tokenizer
        )
        analyzer_no_tokenizer.analyze_sample(conv)

    # Test with tokenizer but token_count=False (should not call tokenizer)
    mock_tokenizer_unused = Mock()
    analyzer_unused = LengthAnalyzer(
        char_count=True,
        word_count=False,
        sentence_count=False,
        token_count=False,  # Token count disabled
        tokenizer=mock_tokenizer_unused,
    )
    conv = _single_message_conversation("Hello, world!")
    message_results, conversation_result = analyzer_unused.analyze_sample(conv)
    # Should not call tokenizer since token_count=False
    mock_tokenizer_unused.encode.assert_not_called()
    # Should still compute char_count
    assert message_results[0].analyzer_metrics["char_count"] == 13


def test_conversation_level_token_count():
    """Test that conversation-level token count is computed correctly with tokenizer."""
    mock_tokenizer = Mock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    # 6 tokens for each message; 10 tokens for conversation
    mock_tokenizer.encode.side_effect = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        list(range(10)),
    ]

    # Create analyzer without dataset
    analyzer = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer,
    )

    # Create a conversation with multiple messages
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello, how are you?"),
            Message(role=Role.ASSISTANT, content="I am doing well, thank you!"),
        ]
    )

    # Analyze the conversation
    message_results, conversation_result = analyzer.analyze_sample(conv)

    # Check that conversation-level token count is computed
    assert "token_count" in conversation_result.analyzer_metrics
    assert conversation_result.analyzer_metrics["token_count"] == 10

    # Verify that apply_chat_template + encode was used for
    # conversation-level token count
    assert mock_tokenizer.apply_chat_template.call_count == 1
    # Two message encodes plus one conversation encode
    assert mock_tokenizer.encode.call_count == 3


def test_conversation_level_token_count_without_dataset():
    """Test that conversation-level token count is computed without a dataset using
    tokenizer chat template directly."""
    mock_tokenizer = Mock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_tokenizer.encode.side_effect = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        list(range(8)),
    ]

    # Create analyzer WITHOUT dataset
    analyzer = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer,
        # No dataset parameter
    )

    # Create a conversation with multiple messages
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello, how are you?"),
            Message(role=Role.ASSISTANT, content="I am doing well, thank you!"),
        ]
    )

    # Analyze the conversation
    message_results, conversation_result = analyzer.analyze_sample(conv)

    # Check that conversation-level token count IS computed
    assert conversation_result.analyzer_metrics["token_count"] == 8
    assert mock_tokenizer.apply_chat_template.call_count == 1
    assert mock_tokenizer.encode.call_count == 3


def test_conversation_level_metrics_aggregation():
    """Test that conversation-level metrics are correctly aggregated from message-level
    metrics."""
    # Test that char, word, and sentence counts are aggregated from message-level
    # results
    mock_tokenizer = Mock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_tokenizer.encode.side_effect = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        list(range(10)),
    ]

    # Create analyzer with all metrics enabled
    analyzer = LengthAnalyzer(
        char_count=True,
        word_count=True,
        sentence_count=True,
        token_count=True,
        tokenizer=mock_tokenizer,
    )

    # Create a conversation with multiple messages
    conv = Conversation(
        messages=[
            Message(
                role=Role.USER, content="Hello, how are you?"
            ),  # 18 chars, 4 words, 1 sentence
            Message(
                role=Role.ASSISTANT, content="I am doing well, thank you!"
            ),  # 26 chars, 6 words, 1 sentence
        ]
    )

    # Analyze the conversation
    message_results, conversation_result = analyzer.analyze_sample(conv)

    # Check conversation-level metrics are aggregated correctly
    assert conversation_result.analyzer_metrics["char_count"] == 46  # 19 + 27
    assert conversation_result.analyzer_metrics["word_count"] == 10  # 4 + 6
    assert conversation_result.analyzer_metrics["sentence_count"] == 2  # 1 + 1
    assert conversation_result.analyzer_metrics["token_count"] == 10
    assert mock_tokenizer.apply_chat_template.call_count == 1
    assert mock_tokenizer.encode.call_count == 3
