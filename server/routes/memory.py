"""FastAPI route definitions for the Memory Environment."""

from __future__ import annotations
from fastapi import APIRouter

from server.schemas.memory import (
    ResetRequest, ResetResponse,
    StepRequest, StepResponse,
    StateResponse, HealthResponse,
)
from server.handlers.memory_handler import handle_reset, handle_step, handle_state

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", environment="memory_env")


@router.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()):
    return handle_reset(
        difficulty=request.difficulty,
        scenario_idx=request.scenario_idx,
    )


@router.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    return handle_step(action=request.action)


@router.get("/state", response_model=StateResponse)
def state():
    return handle_state()
