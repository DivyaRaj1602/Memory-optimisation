"""Episodic Memory — past interaction events stored with timestamps."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import time


@dataclass
class Episode:
    content: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5

    def to_dict(self) -> dict:
        return {"content": self.content, "timestamp": self.timestamp, "importance": self.importance}


@dataclass
class EpisodicMemory:
    capacity: int = 50
    episodes: List[Episode] = field(default_factory=list)

    def store(self, content: str, importance: float = 0.5) -> bool:
        overflow = len(self.episodes) >= self.capacity
        if overflow:
            self.episodes.sort(key=lambda e: e.importance)
            self.episodes.pop(0)
        self.episodes.append(Episode(content=content, importance=importance))
        return not overflow

    def retrieve_recent(self, k: int = 5) -> List[str]:
        sorted_eps = sorted(self.episodes, key=lambda e: e.timestamp, reverse=True)
        return [e.content for e in sorted_eps[:k]]

    def retrieve_all(self) -> List[str]:
        return [e.content for e in self.episodes]

    def clear(self) -> None:
        self.episodes.clear()

    @property
    def usage(self) -> int:
        return len(self.episodes)

    def to_dict(self) -> dict:
        return {
            "episodes": [e.to_dict() for e in self.episodes],
            "usage": self.usage,
            "capacity": self.capacity,
        }
