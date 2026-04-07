"""Pydantic schemas for the Memory Environment API."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ResetRequest(BaseModel):
    difficulty: Optional[str] = None  # "easy" | "medium" | "hard" | None (all)
    scenario_idx: Optional[int] = None


class StepRequest(BaseModel):
    action: int  # discrete action index


class ResetResponse(BaseModel):
    observation: List[float]
    info: Dict[str, Any]


class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    scenario_id: str
    phase: str
    step: int
    difficulty: str
    current_text: str
    done: bool
    memory: Dict[str, Any]
    observation: List[float]


class HealthResponse(BaseModel):
    status: str
    environment: str
