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

from pathlib import Path
from typing import Annotated, Any, Callable, Optional

import typer
from rich.table import Table

from oumi.cli import cli_utils


def judge_dataset_file(
    ctx: typer.Context,
    judge_config: Annotated[
        str,
        typer.Option(
            "--config",
            help="Path to the judge config file",
        ),
    ],
    input_file: Annotated[
        str, typer.Option("--input", help="Path to the dataset input file (jsonl)")
    ],
    output_file: Annotated[
        Optional[str],
        typer.Option("--output", help="Path to the output file (jsonl)"),
    ] = None,
    display_raw_output: bool = False,
):
    """Judge a dataset."""
    # Delayed import
    from oumi import judge

    judge_file(
        ctx=ctx,
        judge_config=judge_config,
        input_file=input_file,
        output_file=output_file,
        display_raw_output=display_raw_output,
        judgment_fn=judge.judge_dataset_file,
    )


def judge_conversations_file(
    ctx: typer.Context,
    judge_config: Annotated[
        str,
        typer.Option(
            "--config",
            help="Path to the judge config file",
        ),
    ],
    input_file: Annotated[
        str, typer.Option("--input", help="Path to the dataset input file (jsonl)")
    ],
    output_file: Annotated[
        Optional[str],
        typer.Option("--output", help="Path to the output file (jsonl)"),
    ] = None,
    display_raw_output: bool = False,
):
    """Judge a list of conversations."""
    # Delayed import
    from oumi import judge

    judge_file(
        ctx=ctx,
        judge_config=judge_config,
        input_file=input_file,
        output_file=output_file,
        display_raw_output=display_raw_output,
        judgment_fn=judge.judge_conversations_file,
    )


def judge_file(
    ctx: typer.Context,
    judge_config: Annotated[
        str,
        typer.Option(
            "--config",
            help="Path to the judge config file",
        ),
    ],
    input_file: Annotated[
        str, typer.Option("--input", help="Path to the dataset input file (jsonl)")
    ],
    output_file: Annotated[
        Optional[str],
        typer.Option("--output", help="Path to the output file (jsonl)"),
    ] = None,
    display_raw_output: bool = False,
    judgment_fn: Callable[..., list[Any]] = ...,
):
    """Judge a dataset or list of conversations."""
    # Delayed import
    from oumi.core.configs.judge_config import JudgeConfig

    # Load configs
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Resolve judge config
    judge_config_obj = JudgeConfig.from_path(path=judge_config, extra_args=extra_args)

    # Ensure the dataset input file exists
    if not Path(input_file).exists():
        typer.echo(f"Input file not found: '{input_file}'")
        raise typer.Exit(code=1)

    # Judge the dataset
    judge_outputs = judgment_fn(
        judge_config=judge_config_obj,
        input_file=input_file,
        output_file=output_file,
    )

    # Calculate the overall score
    overall_score = 0.0
    for judge_output in judge_outputs:
        judgment_score = judge_output.field_scores.get("judgment", None)
        if judgment_score is not None:
            overall_score += judgment_score
        else:
            overall_score = None
            break

    # Display the overall score
    if overall_score is not None:
        overall_score = overall_score / len(judge_outputs)
        cli_utils.CONSOLE.print(
            f"\n[bold blue]Overall Score: {overall_score:.2%}[/bold blue]"
        )

    # Display the judge outputs if no output file was specified
    if not output_file:
        table = Table(
            title="Judge Results",
            title_style="bold magenta",
            show_edge=False,
            show_lines=True,
        )
        table.add_column("Judgment", style="cyan")
        table.add_column("Judgment Score", style="green")
        table.add_column("Explanation", style="yellow")
        if display_raw_output:
            table.add_column("Raw Output", style="white")

        for judge_output in judge_outputs:
            judgment_value = str(judge_output.field_values.get("judgment", "N/A"))
            judgment_score = str(judge_output.field_scores.get("judgment", "N/A"))
            explanation_value = str(judge_output.field_values.get("explanation", "N/A"))

            if display_raw_output:
                table.add_row(
                    judgment_value,
                    judgment_score,
                    explanation_value,
                    judge_output.raw_output,
                )
            else:
                table.add_row(judgment_value, judgment_score, explanation_value)

        cli_utils.CONSOLE.print(table)
    else:
        typer.echo(f"Results saved to {output_file}")
