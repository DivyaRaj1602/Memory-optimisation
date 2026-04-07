"""Builds the observation / state dict consumed by agents."""

from __future__ import annotations
from typing import Dict, List


def build_state(
    current_query: str,
    recent_messages: List[str],
    memory_state: Dict,
    step_number: int,
) -> Dict:
    return {
        "current_query": current_query,
        "recent_messages": list(recent_messages),
        **memory_state,
        "step_number": step_number,
    }
