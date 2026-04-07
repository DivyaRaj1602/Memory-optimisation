"""Evaluates quality of the agent's final response using keyword matching.

Optional: use sentence-transformers for semantic similarity (set USE_EMBEDDINGS=True).
"""

from __future__ import annotations
from typing import List

USE_EMBEDDINGS = False  # flip to True if sentence-transformers is installed
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def grade_response(
    response: str,
    good_keywords: List[str],
    bad_keywords: List[str],
) -> float:
    """
    Returns reward in [-1.0, +1.0].
    """
    resp_lower = response.lower()

    for bk in bad_keywords:
        if bk.lower() in resp_lower:
            return -1.0  # wrong / harmful answer

    if not good_keywords:
        return 0.3  # no strong expectation, generic is fine

    hits = sum(1 for gk in good_keywords if gk.lower() in resp_lower)
    ratio = hits / len(good_keywords)

    if ratio >= 0.8:
        return 1.0
    elif ratio >= 0.4:
        return 0.3 + 0.7 * ratio
    elif ratio > 0:
        return 0.3
    else:
        return -0.2  # nothing matched


def grade_response_semantic(response: str, reference: str) -> float:
    """Optional: cosine similarity based grading."""
    if not USE_EMBEDDINGS:
        return 0.0
    model = _get_model()
    embs = model.encode([response, reference])
    from numpy import dot
    from numpy.linalg import norm
    sim = dot(embs[0], embs[1]) / (norm(embs[0]) * norm(embs[1]) + 1e-8)
    return float(sim)
