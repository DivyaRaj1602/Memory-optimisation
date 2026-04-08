"""
Memory Optimisation — Inference Script
=======================================
Evaluates an LLM agent on the memory management benchmark environment.

Environment variables:
    API_BASE_URL       The API endpoint for the LLM.
                       Default: https://router.huggingface.co/together/v1
    MODEL_NAME         The model identifier to use for inference.
                       Default: Qwen/Qwen2.5-7B-Instruct-Turbo
    HF_TOKEN           Your Hugging Face API key (no default).
    LOCAL_IMAGE_NAME   Docker image name for the environment server.
                       e.g. "memory-env"  (optional — omit to use a running server)

Stdout format (one [START], one [STEP] per step, one [END] per scenario):
    [START] task=<scenario_id> env=memory_env model=<model_name>
    [STEP]  step=<n> action=<action_name> reward=0.00 done=false error=null
    [END]   success=true steps=<n> score=0.00 rewards=r1,r2,...
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Environment variables ────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/together/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # e.g. "memory-env"

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8004")
MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.1   # normalized score in [0, 1]
MAX_REWARD_PER_SCENARIO = 0.75  # theoretical upper bound per scenario

# ── Logging helpers ──────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Environment client ───────────────────────────────────────────────────────

@dataclass
class MemoryObservation:
    current_text: str
    phase: str
    scenario_id: str
    step: int
    difficulty: str
    memory_state: Dict[str, Any]


@dataclass
class MemoryStepResult:
    observation: MemoryObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class MemoryEnvClient:
    """
    Async HTTP client wrapping the Memory Environment FastAPI server.
    Mirrors the OpenEnv reset/step/close interface.
    """

    def __init__(self, server_url: str, container_id: Optional[str] = None):
        self.server_url = server_url.rstrip("/")
        self._container_id = container_id
        self._client = httpx.AsyncClient(timeout=30.0)

    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 8004) -> "MemoryEnvClient":
        """Start the Docker container and wait until /health responds."""
        print(f"[DEBUG] Starting Docker container from image: {image_name}", flush=True)
        result = subprocess.run(
            ["docker", "run", "-d", "-p", f"{port}:{port}", image_name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")

        container_id = result.stdout.strip()
        server_url = f"http://localhost:{port}"
        client = cls(server_url=server_url, container_id=container_id)

        # wait for the server to be ready
        for attempt in range(30):
            try:
                async with httpx.AsyncClient(timeout=2.0) as c:
                    resp = await c.get(f"{server_url}/health")
                    if resp.status_code == 200:
                        print(f"[DEBUG] Server ready after {attempt + 1}s", flush=True)
                        return client
            except Exception:
                pass
            await asyncio.sleep(1)

        raise RuntimeError("Server did not become ready within 30 seconds")

    async def reset(self, scenario_idx: Optional[int] = None, difficulty: Optional[str] = None) -> MemoryStepResult:
        payload: Dict[str, Any] = {}
        if scenario_idx is not None:
            payload["scenario_idx"] = scenario_idx
        if difficulty is not None:
            payload["difficulty"] = difficulty

        resp = await self._client.post(f"{self.server_url}/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()

        obs = MemoryObservation(
            current_text=data["info"].get("current_text", ""),
            phase=data["info"].get("phase", "store"),
            scenario_id=data["info"].get("scenario_id", ""),
            step=data["info"].get("step", 0),
            difficulty=data["info"].get("difficulty", ""),
            memory_state={},
        )
        return MemoryStepResult(observation=obs, reward=0.0, done=False, info=data["info"])

    async def step(self, action: int) -> MemoryStepResult:
        resp = await self._client.post(f"{self.server_url}/step", json={"action": action})
        resp.raise_for_status()
        data = resp.json()

        obs = MemoryObservation(
            current_text=data["info"].get("current_text", ""),
            phase=data["info"].get("phase", "store"),
            scenario_id=data["info"].get("scenario_id", ""),
            step=data["info"].get("step", 0),
            difficulty=data["info"].get("difficulty", ""),
            memory_state={},
        )
        return MemoryStepResult(
            observation=obs,
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )

    async def close(self) -> None:
        await self._client.aclose()
        if self._container_id:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
            )
            print(f"[DEBUG] Stopped container {self._container_id[:12]}", flush=True)


# ── LLM agent ────────────────────────────────────────────────────────────────

ACTION_LIST = [
    "store_working", "store_episodic", "store_preference", "store_intent",
    "store_emotion", "store_personality", "store_fact",
    "retrieve_memory", "discard_memory", "summarize_memory", "do_nothing",
]
ACTION_INDEX = {name: i for i, name in enumerate(ACTION_LIST)}

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI assistant managing a three-layer memory system.
    Given a user message and phase, choose the best memory action.

    Available actions:
    - store_working     : short-term context (capacity: 5)
    - store_episodic    : past events with timestamps (capacity: 50)
    - store_preference  : user preferences, likes, dislikes, dietary restrictions
    - store_intent      : user goals, plans, future actions
    - store_emotion     : user emotional state, feelings, mood
    - store_personality : user personality traits
    - store_fact        : factual info — name, location, job, age
    - retrieve_memory   : search all layers for relevant context (use in query phase)
    - discard_memory    : ignore this message
    - summarize_memory  : condense and store a summary
    - do_nothing        : take no action

    Rules:
    - store phase: pick the best storage action for the message.
    - query phase: always use retrieve_memory.
    - Reply with ONLY the action name. Nothing else.
""").strip()


def get_action_from_llm(
    client: OpenAI,
    phase: str,
    current_text: str,
) -> str:
    """Returns an action name string. Falls back to heuristics on failure."""
    if phase == "query":
        return "retrieve_memory"

    user_prompt = f"Phase: {phase}\nMessage: {current_text}\n\nChoose one action:"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=20,
        )
        raw = (completion.choices[0].message.content or "").strip().lower().replace("-", "_")
        if raw in ACTION_INDEX:
            return raw
        for name in ACTION_LIST:
            if name in raw:
                return name
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)

    return "store_episodic"  # safe fallback


# ── Single scenario runner ───────────────────────────────────────────────────

async def run_scenario(
    env: MemoryEnvClient,
    client: OpenAI,
    scenario_idx: int,
    scenario_id: str,
) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=scenario_id, env="memory_env", model=MODEL_NAME)

    try:
        result = await env.reset(scenario_idx=scenario_idx)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            action_name = get_action_from_llm(client, obs.phase, obs.current_text)
            action_idx = ACTION_INDEX.get(action_name, ACTION_INDEX["store_episodic"])

            try:
                result = await env.step(action_idx)
                reward = result.reward
                done = result.done
                last_error = None
            except Exception as exc:
                reward = 0.0
                done = True
                last_error = str(exc)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_name,
                reward=reward,
                done=done,
                error=last_error,
            )

            if done:
                break

        total_reward = sum(rewards)
        score = max(0.001, min(0.999, total_reward / MAX_REWARD_PER_SCENARIO))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        last_error = str(exc)
        print(f"[DEBUG] Scenario error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN is not set. Add it to your .env file or export it as an environment variable."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # start env from docker image or connect to running server
    if LOCAL_IMAGE_NAME:
        env = await MemoryEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = MemoryEnvClient(server_url=SERVER_URL)

    # all 9 scenarios in fixed order for reproducibility
    from server.db.scenarios import SCENARIOS

    try:
        for idx, scenario in enumerate(SCENARIOS):
            await run_scenario(
                env=env,
                client=client,
                scenario_idx=idx,
                scenario_id=scenario["id"],
            )
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
