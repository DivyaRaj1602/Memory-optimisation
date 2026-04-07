"""Pydantic data models for agent-environment interaction."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class MemoryAction(BaseModel):
    """An action sent by the agent to the environment."""
    action_index: int
    action_name: Optional[str] = None  # populated by the env

    @classmethod
    def from_index(cls, index: int) -> "MemoryAction":
        from server.services.action_handler import ACTION_LIST
        return cls(action_index=index, action_name=ACTION_LIST[index])


class MemoryObservation(BaseModel):
    """The observation returned by the environment after each step."""
    observation: List[float]       # 6-dim numeric vector
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    info: Dict[str, Any] = {}


class MemoryState(BaseModel):
    """Full human-readable state of the environment."""
    scenario_id: str
    phase: str                     # "store" | "query"
    step: int
    difficulty: str
    current_text: str
    done: bool
    memory: Dict[str, Any]         # working / episodic / semantic usage + contents
    observation: List[float]
