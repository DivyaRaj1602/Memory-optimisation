"""Request handlers for the Memory Environment API."""

from __future__ import annotations
from typing import Dict

from server.services.environment import MemoryEnvService
from server.schemas.memory import ResetResponse, StepResponse, StateResponse

# single shared environment instance per server process
_env: MemoryEnvService | None = None


def get_env() -> MemoryEnvService:
    global _env
    if _env is None:
        _env = MemoryEnvService()
    return _env


def handle_reset(difficulty: str | None = None, scenario_idx: int | None = None) -> ResetResponse:
    env = get_env()
    if difficulty is not None:
        env.scenarios = __import__("server.db.scenarios", fromlist=["get_scenarios"]).get_scenarios(difficulty)
        env._scenario_idx = 0
    if scenario_idx is not None:
        env._scenario_idx = scenario_idx
    obs, info = env.reset()
    return ResetResponse(observation=obs.tolist(), info=info)


def handle_step(action: int) -> StepResponse:
    env = get_env()
    obs, reward, done, truncated, info = env.step(action)
    return StepResponse(
        observation=obs.tolist(),
        reward=reward,
        done=done,
        truncated=truncated,
        info=info,
    )


def handle_state() -> StateResponse:
    env = get_env()
    state = env.get_full_state()
    return StateResponse(**state)
