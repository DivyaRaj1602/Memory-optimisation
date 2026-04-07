"""Semantic Memory — structured user model with attribute categories."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

CATEGORIES = ("preferences", "intent", "emotion", "personality", "facts")


@dataclass
class SemanticMemory:
    capacity: int = 30  # total items across all categories
    store_data: Dict[str, List[str]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})

    def store(self, category: str, item: str) -> bool:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Must be one of {CATEGORIES}")
        if self.usage >= self.capacity:
            return False  # overflow
        if item not in self.store_data[category]:
            self.store_data[category].append(item)
        return True

    def retrieve(self, category: str | None = None) -> Dict[str, List[str]] | List[str]:
        if category:
            return list(self.store_data.get(category, []))
        return {k: list(v) for k, v in self.store_data.items()}

    def retrieve_all_flat(self) -> List[str]:
        return [item for items in self.store_data.values() for item in items]

    def clear(self) -> None:
        self.store_data = {c: [] for c in CATEGORIES}

    @property
    def usage(self) -> int:
        return sum(len(v) for v in self.store_data.values())

    def to_dict(self) -> dict:
        return {
            "contents": {k: list(v) for k, v in self.store_data.items()},
            "usage": self.usage,
            "capacity": self.capacity,
        }
