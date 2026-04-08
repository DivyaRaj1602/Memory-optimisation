---
title: Memory Optimisation
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8004
pinned: false
---

# Agent Memory OpenEnv

An OpenEnv-compatible RL environment for training an AI assistant to dynamically manage a three-layer memory system — **working**, **episodic**, and **semantic** — to deliver personalised responses while respecting memory constraints.

## Architecture

```
Conversation Input
       ↓
 FastAPI Server (port 8004)
 /reset  /step  /state  /health
       ↓
 MemoryEnvService (Gymnasium)
       ↓
 MemoryManager
 (working / episodic / semantic)
       ↓
 Agent Action
 (store / retrieve / discard)
       ↓
 Graders
 (storage / retrieval / response)
       ↓
 Reward Function → Next State
```

### Memory Layers

| Layer | Purpose | Capacity |
|-------|---------|----------|
| **Working** | Short-term conversation buffer (FIFO) | 5 items |
| **Episodic** | Past interaction events with importance scores | 50 episodes |
| **Semantic** | Structured user model (preferences, intent, emotion, personality, facts) | 30 items |

### Action Space (11 discrete actions)

`store_working`, `store_episodic`, `store_preference`, `store_intent`, `store_emotion`, `store_personality`, `store_fact`, `retrieve_memory`, `discard_memory`, `summarize_memory`, `do_nothing`

### Reward Components

| Component | Weight | Range |
|-----------|--------|-------|
| Storage correctness | 25% | -0.3 to +0.3 |
| Retrieval relevance | 30% | -0.3 to +0.4 |
| Response quality | 35% | -1.0 to +1.0 |
| Memory efficiency | 10% | -0.2 to 0.0 |

## Project Structure

```
memory-optimisation/
├── inference.py              # Submission inference script (START/STEP/END logs)
├── client.py                 # MemoryEnv (gym.Env) + BaselineAgent + RLAgent
├── models.py                 # Pydantic MemoryAction / MemoryObservation
├── scenario_config.json      # All 9 benchmark scenarios
├── openenv.yaml              # Environment spec
├── pyproject.toml            # Package config
├── requirements.txt          # Full dependencies
├── requirements-server.txt   # Slim deps for Docker server
├── Dockerfile                # FastAPI server on port 8004
├── server/
│   ├── main.py               # FastAPI app entry point
│   ├── routes/memory.py      # /reset /step /state /health /
│   ├── handlers/             # Request handling logic
│   ├── services/             # environment.py, memory_manager.py, action_handler.py
│   ├── memory/               # working.py, episodic.py, semantic.py
│   ├── graders/              # storage_grader.py, retrieval_grader.py, response_grader.py
│   ├── reward/               # reward_function.py
│   ├── schemas/              # Pydantic API schemas
│   └── db/scenarios.py       # 9 built-in scenarios
├── scripts/
│   └── run_llm_baseline.py   # LLM baseline via HF Inference API
└── client_notebooks/
    └── evaluate.ipynb        # Evaluation notebook
```

## Quick Start

### 1. Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the server

```bash
python -m uvicorn server.main:app --host 0.0.0.0 --port 8004 --reload
```

Server runs at `http://localhost:8004`
- `GET  /` — welcome + available endpoints
- `GET  /health` — health check
- `GET  /docs` — interactive API docs
- `POST /reset` — start a new episode
- `POST /step` — take an action `{"action": 0-10}`
- `GET  /state` — current environment state

### 3. Run inference (LLM agent)

In a second terminal:

```bash
# copy and fill in your HF token
cp .env.example .env

python inference.py
```

Output:
```
[START] task=easy_01 env=memory_env model=Qwen/Qwen2.5-7B-Instruct-Turbo
[STEP] step=1 action=store_fact reward=0.08 done=false error=null
[STEP] step=2 action=retrieve_memory reward=0.47 done=true error=null
[END] success=true steps=2 score=0.73 rewards=0.08,0.47
```

### 4. Run LLM baseline evaluation

```bash
python scripts/run_llm_baseline.py
# filter by difficulty
python scripts/run_llm_baseline.py --difficulty easy
```

### 5. Train the RL agent

```bash
python3 -c "from client import train_rl; train_rl(total_timesteps=50000)"
```

Model saves to `models/ppo_memory_agent.zip`.

### 6. Docker

```bash
docker build -t memory-env .
docker run -p 8004:8004 memory-env
```

With Docker, run inference using:
```bash
LOCAL_IMAGE_NAME=memory-env python inference.py
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/together/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-7B-Instruct-Turbo` | Model to use |
| `LOCAL_IMAGE_NAME` | No | — | Docker image name (e.g. `memory-env`) |
