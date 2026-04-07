"""Evaluates whether the agent stored information in the correct memory layer."""

from __future__ import annotations
from typing import List, Tuple


def grade_storage(
    action_name: str,
    content: str,
    ground_truth: List[Tuple[str, str]],
) -> float:
    """
    Args:
        action_name: action the agent took (e.g. "store_preference")
        content: the text being stored
        ground_truth: list of (correct_action, content_keyword) pairs for this message

    Returns:
        reward in [-0.3, +0.3]
    """
    if not ground_truth:
        if action_name.startswith("store_"):
            return -0.1  # trivial store penalty
        return 0.0

    best = -0.3
    content_lower = content.lower()
    for correct_action, keyword in ground_truth:
        kw_lower = keyword.lower()
        content_match = any(w in content_lower for w in kw_lower.split())
        if action_name == correct_action and content_match:
            return 0.3  # perfect
        elif action_name.startswith("store_") and content_match:
            best = max(best, 0.05)  # right info, wrong layer

    if action_name in ("discard_memory", "do_nothing"):
        return -0.3  # should have stored but didn't

    return best
