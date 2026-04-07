"""Evaluates whether the agent retrieved relevant memories for a query."""

from __future__ import annotations
from typing import List


def grade_retrieval(
    retrieved: List[str],
    ground_truth_keywords: List[str],
) -> float:
    """
    Args:
        retrieved: items the agent actually retrieved
        ground_truth_keywords: keywords that should appear in retrieval

    Returns:
        reward in [-0.3, +0.4]
    """
    if not ground_truth_keywords:
        return -0.1 if retrieved else 0.0

    retrieved_text = " ".join(retrieved).lower()

    hits = sum(1 for kw in ground_truth_keywords if kw.lower() in retrieved_text)
    total = len(ground_truth_keywords)

    if hits == 0:
        return -0.3  # missed everything

    recall = hits / total
    reward = -0.3 + 0.7 * recall  # scales from -0.3 → +0.4

    if len(retrieved) > total * 2:
        reward -= 0.1

    return round(max(min(reward, 0.4), -0.3), 3)
