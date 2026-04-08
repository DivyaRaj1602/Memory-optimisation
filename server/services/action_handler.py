"""Translates discrete action indices to named actions and dispatches to MemoryManager."""

from __future__ import annotations
from typing import Dict, Tuple

from server.services.memory_manager import MemoryManager

ACTION_LIST = [
    "store_working",
    "store_episodic",
    "store_preference",
    "store_intent",
    "store_emotion",
    "store_personality",
    "store_fact",
    "retrieve_memory",
    "discard_memory",
    "summarize_memory",
    "do_nothing",
]

NUM_ACTIONS = len(ACTION_LIST)
ACTION_INDEX = {name: i for i, name in enumerate(ACTION_LIST)}


def action_index_to_name(index: int) -> str:
    return ACTION_LIST[index]


def handle_action(manager: MemoryManager, action_index: int, content: str) -> Tuple[str, Dict]:
    action_name = action_index_to_name(action_index)
    info = manager.execute_action(action_name, content)
    return action_name, info
