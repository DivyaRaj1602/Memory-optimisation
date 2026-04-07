"""Memory Environment client.

Provides:
  - MemoryEnv: gym.Env wrapper that calls the FastAPI server (or runs locally).
  - BaselineAgent: heuristic keyword-based agent.
  - RLAgent: Stable-Baselines3 PPO agent wrapper.
  - run_scenario: convenience function to run a single scenario end-to-end.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from server.services.action_handler import ACTION_LIST, NUM_ACTIONS

# ── MemoryEnv ────────────────────────────────────────────────────────────────

class MemoryEnv(gym.Env):
    """
    OpenEnv-compatible gym.Env for memory management.

    Runs the environment locally (embedded server services) without HTTP overhead.
    For HTTP mode, pass use_http=True and server_url.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        difficulty: Optional[str] = None,
        render_mode: Optional[str] = None,
        use_http: bool = False,
        server_url: str = "http://localhost:8004",
    ):
        super().__init__()
        self.render_mode = render_mode
        self._use_http = use_http
        self._server_url = server_url.rstrip("/")

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        if not use_http:
            from server.services.environment import MemoryEnvService
            self._env = MemoryEnvService(difficulty=difficulty, render_mode=render_mode)
        else:
            self._env = None

        self._last_info: Dict[str, Any] = {}

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        if self._use_http:
            return self._http_reset()
        obs, info = self._env.reset(seed=seed, options=options)
        self._last_info = info
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._use_http:
            return self._http_step(action)
        obs, reward, done, truncated, info = self._env.step(action)
        self._last_info = info
        return obs, reward, done, truncated, info

    def render(self):
        if self._env:
            self._env.render()

    # ── HTTP helpers ─────────────────────────────────────────────
    def _http_reset(self) -> Tuple[np.ndarray, Dict]:
        import requests
        resp = requests.post(f"{self._server_url}/reset").json()
        obs = np.array(resp["observation"], dtype=np.float32)
        self._last_info = resp["info"]
        return obs, resp["info"]

    def _http_step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        import requests
        resp = requests.post(f"{self._server_url}/step", json={"action": action}).json()
        obs = np.array(resp["observation"], dtype=np.float32)
        self._last_info = resp["info"]
        return obs, resp["reward"], resp["done"], resp["truncated"], resp["info"]


# ── BaselineAgent ────────────────────────────────────────────────────────────

PREFERENCE_KEYWORDS = [
    "prefer", "like", "love", "enjoy", "favorite", "favourite", "vegetarian",
    "vegan", "allergic", "allergy", "fan of", "into", "sci-fi", "fiction",
]
EMOTION_KEYWORDS = ["feeling", "stressed", "happy", "sad", "anxious", "excited", "angry"]
INTENT_KEYWORDS = ["want to", "planning to", "going to", "need to", "looking for"]
FACT_KEYWORDS = ["live in", "work at", "name is", "born in", "age is", "my name", "i am from"]

ACTION_INDEX = {name: i for i, name in enumerate(ACTION_LIST)}


class BaselineAgent:
    """Rule-based agent: classifies user messages by keywords."""

    def act(self, info: dict) -> int:
        phase = info.get("phase", "store")

        if phase == "query":
            return ACTION_INDEX["retrieve_memory"]

        text = info.get("current_text", "").lower()

        for kw in FACT_KEYWORDS:
            if kw in text:
                return ACTION_INDEX["store_fact"]
        for kw in PREFERENCE_KEYWORDS:
            if kw in text:
                return ACTION_INDEX["store_preference"]
        for kw in EMOTION_KEYWORDS:
            if kw in text:
                return ACTION_INDEX["store_emotion"]
        for kw in INTENT_KEYWORDS:
            if kw in text:
                return ACTION_INDEX["store_intent"]

        return ACTION_INDEX["store_episodic"]


# ── RLAgent ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = Path("models/ppo_memory_agent")


def train_rl(
    total_timesteps: int = 50_000,
    difficulty: Optional[str] = None,
    save_path: Path = DEFAULT_MODEL_PATH,
):
    """Train a PPO agent and save to disk."""
    from stable_baselines3 import PPO

    env = MemoryEnv(difficulty=difficulty)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
    )
    model.learn(total_timesteps=total_timesteps)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {save_path}")
    return model


class RLAgent:
    """Wrapper for SB3 PPO inference."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        from stable_baselines3 import PPO
        self.model = PPO.load(str(model_path))

    def act(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


# ── Scenario runner ──────────────────────────────────────────────────────────

def run_scenario(agent, env: Optional[MemoryEnv] = None, verbose: bool = False) -> Tuple[float, Dict]:
    """
    Run a single episode. Returns (total_reward, final_info).

    agent: BaselineAgent (uses info dict) or RLAgent (uses obs array).
    """
    if env is None:
        env = MemoryEnv()

    obs, info = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        if hasattr(agent, "model"):
            action = agent.act(obs)
        else:
            action = agent.act(info)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if verbose:
            action_name = ACTION_LIST[action]
            print(f"  Step {info['step']:2d} | Phase: {info['phase']:5s} | "
                  f"Action: {action_name:20s} | Reward: {reward:+.4f}")

        if done or truncated:
            break

    return total_reward, info
