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

import importlib.metadata
import importlib.util
import logging
import os
import platform
import sys
import urllib.parse
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import requests
import typer
import yaml
from requests.exceptions import RequestException
from rich.console import Console

from oumi.utils.logging import logger, update_logger_level

CONTEXT_ALLOW_EXTRA_ARGS = {"allow_extra_args": True, "ignore_unknown_options": True}
CONFIG_FLAGS = ["--config", "-c"]
OUMI_FETCH_DIR = "~/.oumi/fetch"
OUMI_GITHUB_RAW = "https://raw.githubusercontent.com/oumi-ai/oumi/main"
_OUMI_PREFIX = "oumi://"

CONSOLE = Console()


def section_header(title, console: Console = CONSOLE):
    """Print a section header with the given title.

    Args:
        title: The title text to display in the header.
        console: The Console object to use for printing.
    """
    console.print(f"\n[blue]{'━' * console.width}[/blue]")
    console.print(f"[yellow]   {title}[/yellow]")
    console.print(f"[blue]{'━' * console.width}[/blue]\n")


def parse_extra_cli_args(ctx: typer.Context) -> list[str]:
    """Parses extra CLI arguments into a list of strings.

    Args:
        ctx: The Typer context object.

    Returns:
        List[str]: The extra CLI arguments
    """
    args = []

    # The following formats are supported:
    # 1. Space separated: "--foo" "2"
    # 2. `=`-separated: "--foo=2"
    try:
        num_args = len(ctx.args)
        idx = 0
        while idx < num_args:
            original_key = ctx.args[idx]
            key = original_key.strip()
            if not key.startswith("--"):
                raise typer.BadParameter(
                    "Extra arguments must start with '--'. "
                    f"Found argument `{original_key}` at position {idx}: `{ctx.args}`"
                )
            # Strip leading "--"

            key = key[2:]
            pos = key.find("=")
            if pos >= 0:
                # '='-separated argument
                value = key[(pos + 1) :].strip()
                key = key[:pos].strip()
                if not key:
                    raise typer.BadParameter(
                        "Empty key name for `=`-separated argument. "
                        f"Found argument `{original_key}` at position {idx}: "
                        f"`{ctx.args}`"
                    )
                idx += 1
            else:
                # Space separated argument
                if idx + 1 >= num_args:
                    raise typer.BadParameter(
                        "Trailing argument has no value assigned. "
                        f"Found argument `{original_key}` at position {idx}: "
                        f"`{ctx.args}`"
                    )
                value = ctx.args[idx + 1].strip()
                idx += 2

            if value.startswith("--"):
                logger.warning(
                    f"Argument value ('{value}') starts with `--`! "
                    f"Key: '{original_key}'"
                )

            cli_arg = f"{key}={value}"
            args.append(cli_arg)
    except ValueError:
        bad_args = " ".join(ctx.args)
        raise typer.BadParameter(
            "Extra arguments must be in `--argname value` pairs. "
            f"Recieved: `{bad_args}`"
        )
    logger.debug(f"\n\nParsed CLI args:\n{args}\n\n")
    return args


def configure_common_env_vars() -> None:
    """Sets common environment variables if needed."""
    if "ACCELERATE_LOG_LEVEL" not in os.environ:
        os.environ["ACCELERATE_LOG_LEVEL"] = "info"
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LogLevel(str, Enum):
    """The available logging levels."""

    DEBUG = logging.getLevelName(logging.DEBUG)
    INFO = logging.getLevelName(logging.INFO)
    WARNING = logging.getLevelName(logging.WARNING)
    ERROR = logging.getLevelName(logging.ERROR)
    CRITICAL = logging.getLevelName(logging.CRITICAL)


def set_log_level(level: Optional[LogLevel]):
    """Sets the logging level for the current command.

    Args:
        level (Optional[LogLevel]): The log level to use.
    """
    if not level:
        return
    uppercase_level = level.upper()
    update_logger_level("oumi", level=uppercase_level)
    CONSOLE.print(f"Set log level to [yellow]{uppercase_level}[/yellow]")


LOG_LEVEL_TYPE = Annotated[
    Optional[LogLevel],
    typer.Option(
        "--log-level",
        "-log",
        help="The logging level for the specified command.",
        show_default=False,
        show_choices=True,
        case_sensitive=False,
        callback=set_log_level,
    ),
]

VERBOSE_TYPE = Annotated[
    bool,
    typer.Option(
        "--verbose",
        "-v",
        help="Enable verbose logging with additional debug information.",
        show_default=True,
    ),
]


def _resolve_oumi_prefix(
    config_path: str, output_dir: Optional[Path] = None
) -> tuple[str, Path]:
    """Resolves oumi:// prefix and determines output directory.

    Args:
        config_path: Path that may contain oumi:// prefix
        output_dir: Optional output directory override

    Returns:
        tuple[str, Path]: (cleaned path, output directory)
    """
    if config_path.lower().startswith(_OUMI_PREFIX):
        config_path = config_path[len(_OUMI_PREFIX) :]

    config_dir = output_dir or os.environ.get("OUMI_DIR") or OUMI_FETCH_DIR
    config_dir = Path(config_dir).expanduser()
    config_dir.mkdir(parents=True, exist_ok=True)

    return config_path, config_dir


def resolve_and_fetch_config(
    config_path: str, output_dir: Optional[Path] = None, force: bool = True
) -> Path:
    """Resolve oumi:// prefix and fetch config if needed.

    Args:
        config_path: Original config path that may contain oumi:// prefix
        output_dir: Optional override for output directory
        force: Whether to overwrite an existing config

    Returns:
        Path: Local path to the config file
    """
    if not config_path.lower().startswith(_OUMI_PREFIX):
        return Path(config_path)

    # Remove oumi:// prefix if present
    new_config_path, config_dir = _resolve_oumi_prefix(config_path, output_dir)

    try:
        # Check destination first
        local_path = (config_dir or Path(OUMI_FETCH_DIR).expanduser()) / new_config_path
        if local_path.exists() and not force:
            msg = f"Config already exists at {local_path}. Use --force to overwrite"
            logger.error(msg)
            raise RuntimeError(msg)

        # Fetch from GitHub
        github_url = f"{OUMI_GITHUB_RAW}/{new_config_path.lstrip('/')}"
        response = requests.get(github_url)
        response.raise_for_status()
        config_content = response.text

        # Validate YAML
        yaml.safe_load(config_content)

        # Save to destination
        if local_path.exists():
            logger.warning(f"Overwriting existing config at {local_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "w") as f:
            f.write(config_content)
        logger.info(f"Successfully downloaded config to {local_path}")
    except RequestException as e:
        logger.error(f"Failed to download config from GitHub: {e}")
        raise
    except yaml.YAMLError:
        logger.error("Invalid YAML configuration")
        raise

    return Path(local_path)


def create_github_issue_url(exception: Exception, traceback_str: str) -> str:
    """Create a prefilled GitHub issue URL aligned with the bug report template.

    Args:
        exception: The exception that occurred
        traceback_str: The full traceback string

    Returns:
        str: URL for creating a prefilled GitHub issue
    """
    # 2000 is the max length of a URL to ensure it works with any browser
    MAX_URL_LENGTH = 2000
    base_url = "https://github.com/oumi-ai/oumi/issues/new?"

    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    system_info = f"{platform.system()} {platform.release()}"
    oumi_version = importlib.metadata.version("oumi")
    system_info_content = (
        f"**Please paste the output of `oumi env` here.** "
        f"If you can't run this command, here's basic system info:\n\n"
        f"- Operating system: {system_info}\n"
        f"- Python version: {python_version}\n"
        f"- Oumi version: {oumi_version}"
    )

    title_template = "[Automatic Error Report]: {error_msg}"
    reproduction_template = (
        "Steps to reproduce:\n\n"
        "Command executed: `{command}`\n\n"
        "Stack trace:\n```\n{traceback}\n```"
    )

    exception_str = str(exception)
    command_str = " ".join(sys.argv)

    # Limit the title to 50 chars (unencoded)
    title_error = exception_str[:50]

    command_part = command_str
    error_part = exception_str
    stack_part = traceback_str or ""

    def build_url(command_part: str, error_part: str, stack_part: str) -> str:
        params = {
            "template": "bug-report.yaml",
            "title": title_template.format(error_msg=title_error),
            "what-happened": error_part,
            "reproduction-steps": reproduction_template.format(
                command=command_part, traceback=stack_part
            ),
            "system-info": system_info_content,
        }
        return f"{base_url}{urllib.parse.urlencode(params)}"

    full_url = build_url(command_part, error_part, stack_part)

    # If too long, iteratively shrink fields by encoded length
    while len(full_url) > MAX_URL_LENGTH:
        if stack_part:
            # shrink stack from the front (keep tail)
            cut = max(1, len(stack_part) // 2)
            stack_part = stack_part[cut:]
        elif error_part:
            cut = max(1, len(error_part) // 2)
            error_part = error_part[:-cut]
        elif command_part:
            cut = max(1, len(command_part) // 2)
            command_part = command_part[:-cut]
        else:
            return f"{base_url}template=bug-report.yaml"
        full_url = build_url(command_part, error_part, stack_part)
    return full_url
