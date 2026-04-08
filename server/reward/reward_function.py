"""Combines partial rewards from all graders into a single scalar reward."""

from __future__ import annotations
from typing import Dict, List, Tuple

from server.graders.storage_grader import grade_storage
from server.graders.retrieval_grader import grade_retrieval
from server.graders.response_grader import grade_response


# reward component weights
W_STORAGE = 0.25
W_RETRIEVAL = 0.30
W_RESPONSE = 0.35
W_EFFICIENCY = 0.10


def compute_reward(
    action_name: str,
    content: str,
    storage_ground_truth: List[Tuple[str, str]],
    retrieved: List[str],
    retrieval_ground_truth: List[str],
    response_text: str,
    good_keywords: List[str],
    bad_keywords: List[str],
    memory_usage: Dict[str, int],
    memory_caps: Dict[str, int],
    phase: str = "store",  # "store" or "query"
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (total_reward, breakdown_dict).
    """
    breakdown: Dict[str, float] = {}

    if phase == "store":
        r_store = grade_storage(action_name, content, storage_ground_truth)
        breakdown["storage"] = r_store
        total_use = sum(memory_usage.values())
        total_cap = sum(memory_caps.values())
        eff_penalty = -0.2 if total_use > total_cap * 0.9 else 0.0
        breakdown["efficiency"] = eff_penalty
        total = W_STORAGE * r_store + W_EFFICIENCY * eff_penalty
    else:
        r_ret = grade_retrieval(retrieved, retrieval_ground_truth)
        r_resp = grade_response(response_text, good_keywords, bad_keywords)
        breakdown["retrieval"] = r_ret
        breakdown["response"] = r_resp
        total = W_RETRIEVAL * r_ret + W_RESPONSE * r_resp

    # Normalize raw reward to (0, 1) exclusive — OpenEnv requires task scores
    # strictly between 0 and 1. Raw rewards span roughly [-0.5, 0.5], so
    # shifting by 0.5 maps that range to [0, 1], then we clamp away from edges.
    score = round(max(0.001, min(0.999, total + 0.5)), 4)
    breakdown["total"] = score
    return score, breakdown
