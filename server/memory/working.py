"""Working Memory — short-term conversation context (FIFO buffer)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class WorkingMemory:
    capacity: int = 5
    buffer: List[str] = field(default_factory=list)

    def store(self, item: str) -> bool:
        """Store item. Returns False if overflow (oldest evicted)."""
        overflow = len(self.buffer) >= self.capacity
        if overflow:
            self.buffer.pop(0)
        self.buffer.append(item)
        return not overflow

    def retrieve_all(self) -> List[str]:
        return list(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()

    @property
    def usage(self) -> int:
        return len(self.buffer)

    def to_dict(self) -> dict:
        return {"items": list(self.buffer), "usage": self.usage, "capacity": self.capacity}
