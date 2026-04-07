"""Unified memory manager — coordinates all three layers."""

from __future__ import annotations
from typing import Dict, List

from server.memory.working import WorkingMemory
from server.memory.episodic import EpisodicMemory
from server.memory.semantic import SemanticMemory


# maps action names → semantic category
ACTION_TO_CATEGORY = {
    "store_preference": "preferences",
    "store_intent": "intent",
    "store_emotion": "emotion",
    "store_personality": "personality",
    "store_fact": "facts",
}


class MemoryManager:
    def __init__(self, working_cap: int = 5, episodic_cap: int = 50, semantic_cap: int = 30):
        self.working = WorkingMemory(capacity=working_cap)
        self.episodic = EpisodicMemory(capacity=episodic_cap)
        self.semantic = SemanticMemory(capacity=semantic_cap)
        self._last_retrieved: List[str] = []

    def execute_action(self, action: str, content: str) -> Dict:
        """Execute a memory action. Returns info dict with success/overflow flags."""
        info = {"action": action, "content": content, "success": True, "overflow": False}

        if action == "store_working":
            ok = self.working.store(content)
            info["overflow"] = not ok

        elif action == "store_episodic":
            ok = self.episodic.store(content)
            info["overflow"] = not ok

        elif action in ACTION_TO_CATEGORY:
            category = ACTION_TO_CATEGORY[action]
            ok = self.semantic.store(category, content)
            if not ok:
                info["success"] = False
                info["overflow"] = True

        elif action == "retrieve_memory":
            self._last_retrieved = self._retrieve_relevant(content)
            info["retrieved"] = self._last_retrieved

        elif action == "discard_memory":
            info["discarded"] = True

        elif action == "summarize_memory":
            summary = content[:80] + "..." if len(content) > 80 else content
            self.episodic.store(f"[summary] {summary}")

        elif action == "do_nothing":
            pass

        return info

    def _retrieve_relevant(self, query: str) -> List[str]:
        """Naive keyword retrieval across all layers."""
        query_lower = query.lower()
        results = []
        for item in self.working.retrieve_all():
            if any(w in item.lower() for w in query_lower.split()):
                results.append(item)
        for item in self.episodic.retrieve_all():
            if any(w in item.lower() for w in query_lower.split()):
                results.append(item)
        for item in self.semantic.retrieve_all_flat():
            if any(w in item.lower() for w in query_lower.split()):
                results.append(item)
        return results

    @property
    def last_retrieved(self) -> List[str]:
        return self._last_retrieved

    def reset(self) -> None:
        self.working.clear()
        self.episodic.clear()
        self.semantic.clear()
        self._last_retrieved = []

    def get_state(self) -> Dict:
        return {
            "working_memory_usage": self.working.usage,
            "episodic_memory_usage": self.episodic.usage,
            "semantic_memory_usage": self.semantic.usage,
            "memory_contents": self.semantic.retrieve(),
            "retrieved_memories": list(self._last_retrieved),
        }
