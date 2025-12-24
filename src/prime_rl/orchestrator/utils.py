from typing import Any, AsyncContextManager

import logfire
import pandas as pd
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from rich.console import Console
from rich.table import Table
import verifiers as vf
from verifiers.utils.async_utils import maybe_semaphore

from prime_rl.orchestrator.config import SamplingConfig
from prime_rl.transport.types import TrainingSample
from prime_rl.utils.utils import (
    format_num,
    format_time,
)

SEMAPHORE: AsyncContextManager | None = None


async def set_semaphore(limit: int):
    global SEMAPHORE
    SEMAPHORE = await maybe_semaphore(limit)


async def get_semaphore() -> AsyncContextManager:
    global SEMAPHORE
    assert SEMAPHORE is not None, "Semaphore not set"
    return SEMAPHORE


def get_sampling_args(sampling_config: SamplingConfig) -> dict:
    # Convert SamplingConfig to vLLM OAI sampling args
    # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters_2
    sampling_args = dict(sampling_config)
    sampling_args["top_p"] = 1.0
    sampling_args["logprobs"] = True
    sampling_args["extra_body"] = {
        **sampling_config.extra_body,
        "return_token_ids": True,  # Always return token IDs
        "prompt_logprobs": True,  # Always return prompt logprobs
        "top_k": -1,
        "min_p": 0.0,
    }
    sampling_args["extra_body"]["min_tokens"] = sampling_args.pop("min_tokens")
    sampling_args["extra_body"]["repetition_penalty"] = sampling_args.pop("repetition_penalty")
    return sampling_args


def parse_num_completion_tokens(responses: list[list[ChatCompletion]]) -> list[int]:
    """Parses the number of tokens from a list of chat completions returned by OAI API."""
    all_num_completion_tokens = []
    for response in responses:
        num_completion_tokens = 0
        for chat_completion in response:
            assert isinstance(chat_completion, ChatCompletion)
            assert chat_completion.usage is not None, "Usage should be present in the response"
            usage = chat_completion.usage
            assert isinstance(usage, CompletionUsage)
            num_completion_tokens += usage.completion_tokens
        all_num_completion_tokens.append(num_completion_tokens)
    assert len(all_num_completion_tokens) == len(responses), (
        "Number of completion tokens should be the same as the number of responses"
    )
    return all_num_completion_tokens


def parse_is_truncated_completions(responses: list[list[ChatCompletion]]) -> list[bool]:
    """Parses whether the completions were truncated from a list of (multi-turn) OAI chat completions"""
    all_is_truncated = []
    for response in responses:
        is_truncated = False
        for chat_completion in response:
            assert isinstance(chat_completion, ChatCompletion)
            assert len(chat_completion.choices) == 1, "Response should always have one choice"
            choice = chat_completion.choices[0]
            assert isinstance(choice, Choice)
            if choice.finish_reason == "length":
                is_truncated = True
        all_is_truncated.append(is_truncated)
    return all_is_truncated


def print_benchmark(history: dict[str, list[Any]]) -> None:
    """
    Print benchmark results as rich table. Shows formatted values for the
    inference throughput and overall step time. First first N rows show the
    per-step values, and the last row shows the mean, std, min, and max values.
    """
    history.pop("step")
    assert all(len(v) for v in history.values()), "All metrics must have logged the same number of steps"

    # Turn metric history into pd.DataFrame
    df = pd.DataFrame(dict(history.items()))
    columns = {
        "perf/throughput": "Throughput",
        "time/step": "Step Time",
    }
    df = df.rename(columns=columns)
    df = df[list(columns.values())]
    df = df.iloc[1:]  # Exclude first row

    # Setup console
    console = Console()
    table = Table(title="Benchmark")

    # Add columns
    table.add_column("Step", justify="right")
    for col in df.columns:
        table.add_column(col, justify="center", style="magenta")

    # Add formatted rows
    formatted_df = pd.DataFrame(columns=df.columns)
    formatted_df["Step Time"] = df["Step Time"].apply(format_time)
    formatted_df["Throughput"] = df["Throughput"].apply(format_num, precision=2)
    for step, row in formatted_df.iterrows():
        table.add_row(*([str(step)] + [str(x) for x in row]))

    # Separator
    num_table_columns = 1 + len(df.columns)
    table.add_row(*([""] * num_table_columns))

    # Add row for formatted, aggregated statistics
    mean_df = df.describe().loc[["mean", "std", "min", "max"], :]
    formatted_mean_df = pd.DataFrame(columns=mean_df.columns)
    formatted_mean_df["Step Time"] = mean_df["Step Time"].apply(format_time)
    formatted_mean_df["Throughput"] = mean_df["Throughput"].apply(format_num, precision=2)
    mean_row = ["Overall"] + formatted_mean_df.T.apply(
        lambda row: f"{row['mean']} ± {row['std']} [{row['min']}, {row['max']}]", axis=1
    ).tolist()
    table.add_row(*mean_row)

    # Display table
    console.print(table)

@logfire.instrument()
def logfire_log_training_sample(rollout: vf.State, examples: list[TrainingSample], pair_index: int, total_pairs: int) -> None:
    """Log a State -> TrainingSample pair to logfire with nicely formatted output."""
    from prime_rl.utils.vf import get_completion_len, get_prompt_len, get_seq_len

    trajectory = rollout.get("trajectory", [])
    try:
        prompt_len = get_prompt_len(rollout) if trajectory else 0
        completion_len = get_completion_len(rollout) if trajectory else 0
        seq_len = get_seq_len(rollout) if trajectory else 0
    except (KeyError, IndexError):
        prompt_len = 0
        completion_len = 0
        seq_len = 0

    formatted_rollout = {
        "example_id": rollout.get("example_id"),
        "task": rollout.get("task"),
        "reward": rollout.get("reward"),
        "is_truncated": rollout.get("is_truncated"),
        "error": type(rollout.get("error")).__name__ if rollout.get("error") is not None else None,
        "num_turns": len(trajectory),
        "prompt_len": prompt_len,
        "completion_len": completion_len,
        "seq_len": seq_len,
    }

    formatted_trajectory = []
    for step_idx, step in enumerate(trajectory):
        if not isinstance(step, dict):
            continue
        step_tokens = step.get("tokens")
        prompt_messages = step.get("prompt")
        completion_messages = step.get("completion")
        
        formatted_step = {
            "step_index": step_idx,
            "reward": step.get("reward"),
            "advantage": step.get("advantage"),
            "has_tokens": step_tokens is not None,
        }
        
        if prompt_messages:
            formatted_step["prompt_messages"] = prompt_messages
        if completion_messages:
            formatted_step["completion_messages"] = completion_messages
        
        if step_tokens and isinstance(step_tokens, dict):
            formatted_step.update({
                "prompt_token_count": len(step_tokens.get("prompt_ids", [])),
                "completion_token_count": len(step_tokens.get("completion_ids", [])),
                "has_logprobs": len(step_tokens.get("completion_logprobs", [])) > 0,
            })
        formatted_trajectory.append(formatted_step)
    formatted_rollout["trajectory_steps"] = formatted_trajectory

    formatted_examples = []
    for j, example in enumerate(examples):
        formatted_examples.append({
            "example_index": j,
            "prompt_len": len(example.prompt_ids) if example.prompt_ids else 0,
            "completion_len": len(example.completion_ids) if example.completion_ids else 0,
            "advantage": example.advantage,
            "has_logprobs": len(example.completion_logprobs) > 0 if example.completion_logprobs else False,
        })

    logfire.info(
        f"State to TrainingSample pair {pair_index}/{total_pairs}",
        rollout=formatted_rollout,
        training_examples=formatted_examples,
        num_examples_from_rollout=len(examples),
    )
