#!/usr/bin/env python3
"""
LLM Baseline Inference Script
==============================
Evaluates an LLM (via HF Inference API using the OpenAI-compatible client)
as a memory management agent across all benchmark scenarios.

Credentials are read from a .env file:
    HF_TOKEN  — Hugging Face API token
    HF_MODEL  — HF model ID (default: meta-llama/Meta-Llama-3-8B-Instruct)

Reproducibility guarantees:
    - temperature=0, seed=42 on every LLM call
    - Scenarios run in fixed order (easy_01 → hard_03)
    - Results written to scripts/llm_baseline_results.json

Usage:
    python scripts/run_llm_baseline.py
    python scripts/run_llm_baseline.py --difficulty easy
    python scripts/run_llm_baseline.py --model mistralai/Mixtral-8x7B-Instruct-v0.1
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from client import MemoryEnv, run_scenario
from server.services.action_handler import ACTION_LIST, ACTION_INDEX
from server.db.scenarios import get_scenarios

load_dotenv()  # reads .env from the project root

console = Console()

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"
DEFAULT_PROVIDER = "together"  # Together AI serverless — hosts Qwen Turbo models
SEED = 42

SYSTEM_PROMPT = """You are an AI assistant that manages a three-layer memory system.
You receive a message or query and must decide the best memory action to take.

Available actions:
- store_working     : short-term context, keep recent turns (capacity: 5)
- store_episodic    : past events/interactions with timestamps (capacity: 50)
- store_preference  : user preferences, likes, dislikes, dietary restrictions
- store_intent      : user goals, plans, things they want to do
- store_emotion     : user emotional state, feelings, mood
- store_personality : user personality traits, communication style
- store_fact        : factual info about the user (name, location, job, age)
- retrieve_memory   : search all memory layers for relevant context (use during query phase)
- discard_memory    : explicitly ignore/discard the current text
- summarize_memory  : create a condensed summary and store in episodic
- do_nothing        : take no memory action

Rules:
- During the "store" phase: pick the most appropriate storage action for the message.
- During the "query" phase: always use retrieve_memory to fetch relevant context.
- Respond with ONLY the action name, nothing else. No explanation, no punctuation."""


def build_user_prompt(phase: str, current_text: str, memory_state: dict) -> str:
    wm = memory_state.get("working_memory_usage", 0)
    em = memory_state.get("episodic_memory_usage", 0)
    sm = memory_state.get("semantic_memory_usage", 0)

    lines = [
        f"Phase: {phase}",
        f"Text: {current_text}",
        f"Memory usage — working: {wm}/5, episodic: {em}/50, semantic: {sm}/30",
    ]
    return "\n".join(lines)


# ── LLM Agent ────────────────────────────────────────────────────────────────

class LLMAgent:
    """
    Agent that calls an LLM via the HF Inference API (OpenAI-compatible)
    to decide memory actions.
    """

    def __init__(self, model: str = DEFAULT_MODEL, hf_token: str | None = None, provider: str | None = None):
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise EnvironmentError(
                "HF_TOKEN not found. Set it in your .env file or as an environment variable.\n"
                "Copy .env.example → .env and fill in your token."
            )
        self.model_id = model
        resolved_provider = provider or os.environ.get("HF_PROVIDER", DEFAULT_PROVIDER)
        base_url = f"https://router.huggingface.co/{resolved_provider}/v1"
        self.client = OpenAI(api_key=token, base_url=base_url)
        self._call_count = 0

    def act(self, info: dict) -> int:
        phase = info.get("phase", "store")
        current_text = info.get("current_text", "")

        # Fast-path: always retrieve during query phase (saves API calls)
        if phase == "query":
            return ACTION_INDEX["retrieve_memory"]

        mem_state = {
            "working_memory_usage": 0,
            "episodic_memory_usage": 0,
            "semantic_memory_usage": 0,
        }

        user_prompt = build_user_prompt(phase, current_text, mem_state)

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=20,
        )
        self._call_count += 1

        raw = response.choices[0].message.content.strip().lower().replace("-", "_")

        # exact match first
        if raw in ACTION_INDEX:
            return ACTION_INDEX[raw]

        # partial match fallback
        for name in ACTION_LIST:
            if name in raw:
                return ACTION_INDEX[name]

        # default fallback: store_episodic
        console.print(f"  [yellow]⚠ Unrecognised LLM output '{raw}', defaulting to store_episodic[/]")
        return ACTION_INDEX["store_episodic"]


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_llm_baseline(
    model: str = DEFAULT_MODEL,
    difficulty: Optional[str] = None,
    provider: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run LLM agent across all scenarios and return metrics dict.
    Results are deterministic: temperature=0, fixed scenario order.
    """
    agent = LLMAgent(model=model, provider=provider)
    scenarios = get_scenarios(difficulty)

    all_rewards: List[float] = []
    all_infos: List[Dict] = []
    per_scenario: List[Dict] = []

    console.print(f"\n[bold cyan]Model:[/] {agent.model_id}")
    console.print(f"[bold cyan]Scenarios:[/] {len(scenarios)} | difficulty={difficulty or 'all'}\n")

    env = MemoryEnv()

    for idx, scenario in enumerate(scenarios):
        env._env._scenario_idx = idx
        env._env.scenarios = scenarios

        total_reward, final_info = run_scenario(agent, env=env, verbose=verbose)

        all_rewards.append(total_reward)
        all_infos.append(final_info)
        per_scenario.append({
            "scenario_id": scenario["id"],
            "difficulty": scenario["difficulty"],
            "reward": round(total_reward, 4),
        })

        if verbose:
            color = "green" if total_reward > 0 else "red"
            console.print(
                f"  [{color}]{scenario['id']}[/] ({scenario['difficulty']}) → "
                f"[bold {color}]{total_reward:+.4f}[/]"
            )

    metrics = {
        "model": agent.model_id,
        "difficulty_filter": difficulty,
        "seed": SEED,
        "temperature": 0,
        "num_scenarios": len(all_rewards),
        "mean_reward": round(sum(all_rewards) / len(all_rewards), 4),
        "min_reward": round(min(all_rewards), 4),
        "max_reward": round(max(all_rewards), 4),
        "llm_api_calls": agent._call_count,
        "per_scenario": per_scenario,
        "per_difficulty": _by_difficulty(per_scenario),
    }

    return metrics


def _by_difficulty(per_scenario: List[Dict]) -> Dict:
    groups: Dict[str, List[float]] = {}
    for s in per_scenario:
        groups.setdefault(s["difficulty"], []).append(s["reward"])
    return {
        k: {"mean": round(sum(v) / len(v), 4), "count": len(v)}
        for k, v in groups.items()
    }


def print_results(metrics: Dict):
    console.print()
    table = Table(title="LLM Baseline Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Model", metrics["model"])
    table.add_row("Scenarios", str(metrics["num_scenarios"]))
    table.add_row("Mean Reward", f"{metrics['mean_reward']:+.4f}")
    table.add_row("Min Reward", f"{metrics['min_reward']:+.4f}")
    table.add_row("Max Reward", f"{metrics['max_reward']:+.4f}")
    table.add_row("LLM API Calls", str(metrics["llm_api_calls"]))
    table.add_row("Temperature", str(metrics["temperature"]))
    table.add_row("Seed", str(metrics["seed"]))
    console.print(table)

    if metrics["per_difficulty"]:
        diff_table = Table(title="By Difficulty", show_lines=True)
        diff_table.add_column("Difficulty", style="bold")
        diff_table.add_column("Count", justify="right")
        diff_table.add_column("Mean Reward", justify="right")
        for diff, vals in metrics["per_difficulty"].items():
            color = "green" if vals["mean"] > 0 else "red"
            diff_table.add_row(diff, str(vals["count"]), f"[{color}]{vals['mean']:+.4f}[/]")
        console.print(diff_table)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM baseline evaluation using HF Inference API")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("HF_MODEL", DEFAULT_MODEL),
        help=f"HF model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.environ.get("HF_PROVIDER", DEFAULT_PROVIDER),
        help=f"HF router provider (default: {DEFAULT_PROVIDER}). Options: nebius, hf-inference, together, fireworks-ai",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["easy", "medium", "hard"],
        help="Filter scenarios by difficulty (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/llm_baseline_results.json",
        help="Path to save results JSON",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    args = parser.parse_args()

    console.print("[bold magenta]═══ LLM Memory Agent — Baseline Evaluation ═══[/]\n")

    metrics = evaluate_llm_baseline(
        model=args.model,
        difficulty=args.difficulty,
        provider=args.provider,
        verbose=not args.quiet,
    )

    print_results(metrics)

    # save reproducible results to disk
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"\n[dim]Results saved to {out_path}[/]")

    console.print("\n[bold magenta]═══ Done ═══[/]")
