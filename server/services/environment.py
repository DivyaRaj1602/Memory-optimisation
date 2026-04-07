"""Core OpenEnv-compatible Gymnasium environment for memory management RL."""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List, Tuple

from server.services.memory_manager import MemoryManager
from server.services.state_builder import build_state
from server.services.action_handler import ACTION_LIST, NUM_ACTIONS, handle_action
from server.reward.reward_function import compute_reward
from server.db.scenarios import get_scenarios


class MemoryEnvService(gym.Env):
    """
    Observation: flattened numeric vector encoding memory state.
    Action: discrete index into ACTION_LIST.

    Episode flow per scenario:
      Phase 1 (store): one step per user message → agent picks storage action.
      Phase 2 (query): agent retrieves + generates response → gets query reward.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, difficulty: str | None = None, render_mode: str | None = None):
        super().__init__()
        self.scenarios = get_scenarios(difficulty)
        self.render_mode = render_mode

        # observation: [wm_usage, em_usage, sm_usage, step, num_retrieved, msg_len_norm]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.manager = MemoryManager()
        self._scenario: Dict = {}
        self._message_idx = 0
        self._phase = "store"
        self._step_count = 0
        self._done = False
        self._scenario_idx = 0

    # ── gym interface ───────────────────────────────────────────
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.manager.reset()
        self._scenario = self.scenarios[self._scenario_idx % len(self.scenarios)]
        self._scenario_idx += 1
        self._message_idx = 0
        self._phase = "store"
        self._step_count = 0
        self._done = False
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        scenario = self._scenario
        reward = 0.0
        info: Dict[str, Any] = {}

        if self._phase == "store":
            msg = scenario["messages"][self._message_idx]
            action_name, act_info = handle_action(self.manager, action, msg)
            info["action_info"] = act_info

            gt = [
                (gt_action, gt_content)
                for (midx, gt_action, gt_content) in scenario["ground_truth_storage"]
                if midx == self._message_idx
            ]

            mem_state = self.manager.get_state()
            reward, breakdown = compute_reward(
                action_name=action_name,
                content=msg,
                storage_ground_truth=gt,
                retrieved=[],
                retrieval_ground_truth=[],
                response_text="",
                good_keywords=[],
                bad_keywords=[],
                memory_usage={
                    "working": mem_state["working_memory_usage"],
                    "episodic": mem_state["episodic_memory_usage"],
                    "semantic": mem_state["semantic_memory_usage"],
                },
                memory_caps={"working": 5, "episodic": 50, "semantic": 30},
                phase="store",
            )
            info["reward_breakdown"] = breakdown

            self._message_idx += 1
            if self._message_idx >= len(scenario["messages"]):
                self._phase = "query"

        elif self._phase == "query":
            query = scenario["query"]
            action_name, act_info = handle_action(self.manager, action, query)
            info["action_info"] = act_info

            retrieved = self.manager.last_retrieved
            response_text = f"Based on what I know: {', '.join(retrieved)}" if retrieved else "I don't have enough information."

            reward, breakdown = compute_reward(
                action_name=action_name,
                content=query,
                storage_ground_truth=[],
                retrieved=retrieved,
                retrieval_ground_truth=scenario["ground_truth_retrieval"],
                response_text=response_text,
                good_keywords=scenario["good_response_keywords"],
                bad_keywords=scenario["bad_response_keywords"],
                memory_usage={
                    "working": self.manager.get_state()["working_memory_usage"],
                    "episodic": self.manager.get_state()["episodic_memory_usage"],
                    "semantic": self.manager.get_state()["semantic_memory_usage"],
                },
                memory_caps={"working": 5, "episodic": 50, "semantic": 30},
                phase="query",
            )
            info["reward_breakdown"] = breakdown
            info["response"] = response_text
            self._done = True

        self._step_count += 1
        obs = self._get_obs()
        info.update(self._get_info())

        return obs, reward, self._done, False, info

    # ── helpers ─────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        ms = self.manager.get_state()
        return np.array([
            ms["working_memory_usage"] / 5.0,
            ms["episodic_memory_usage"] / 50.0,
            ms["semantic_memory_usage"] / 30.0,
            self._step_count / 20.0,
            len(ms["retrieved_memories"]) / 10.0,
            len(self._current_text()) / 200.0,
        ], dtype=np.float32).clip(0, 1)

    def _current_text(self) -> str:
        s = self._scenario
        if self._phase == "store" and self._message_idx < len(s["messages"]):
            return s["messages"][self._message_idx]
        return s.get("query", "")

    def _get_info(self) -> Dict:
        return {
            "scenario_id": self._scenario.get("id", ""),
            "phase": self._phase,
            "step": self._step_count,
            "difficulty": self._scenario.get("difficulty", ""),
            "current_text": self._current_text(),
        }

    def get_full_state(self) -> Dict:
        """Returns the full serializable state for the /state endpoint."""
        ms = self.manager.get_state()
        info = self._get_info()
        return {
            **info,
            "done": self._done,
            "memory": ms,
            "observation": self._get_obs().tolist(),
        }

    def render(self):
        if self.render_mode == "human":
            info = self._get_info()
            ms = self.manager.get_state()
            print(f"\n[Step {info['step']}] Phase: {info['phase']} | "
                  f"Scenario: {info['scenario_id']} | Text: {info['current_text']}")
            print(f"  Memory → WM:{ms['working_memory_usage']} EM:{ms['episodic_memory_usage']} "
                  f"SM:{ms['semantic_memory_usage']}")
            print(f"  Semantic contents: {ms['memory_contents']}")
