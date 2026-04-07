# 🧠 Agent Memory OpenEnv

An RL environment for training an AI assistant to dynamically manage a three-layer memory system — **working**, **episodic**, and **semantic** — to deliver personalized responses while respecting memory constraints.

## Architecture

```
Conversation Input
       ↓
 MemoryEnv (Gymnasium)
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

## Quick Start

### 1. Install

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Baseline Evaluation

```bash
python scripts/run_baseline.py
```

This runs the heuristic keyword-matching agent across all 9 built-in scenarios (easy, medium, hard) and prints per-step rewards plus a summary table.

### 3. Train the RL Agent

```bash
python -c "from agents.rl_agent import train; train(total_timesteps=50000)"
```

The trained PPO model saves to `models/ppo_memory_agent.zip`.

### 4. Evaluate the RL Agent

```bash
python evaluation/evaluate.py --agent rl --model-path models/ppo_memory_agent
```

### 5. Launch Interactive Demo

```bash
python app/app.py
```

Opens a Gradio UI at `http://localhost:7860` where you can step through scenarios manually or let the baseline agent auto-play.

### 6. Docker

```bash
docker build -t memory-env .
docker run -p 7860:7860 memory-env
```

## API Keys Required

| What | Required? | Notes |
|------|-----------|-------|
| **None (default)** | ✅ No keys | Baseline + PPO training + keyword grading all run locally |
| **HuggingFace token** | Optional | Only if you use gated models (e.g. Mistral) for response generation |
| **sentence-transformers** | Optional | Set `USE_EMBEDDINGS=True` in `graders/response_grader.py` for semantic grading — downloads `all-MiniLM-L6-v2` automatically, no key needed |

## Project Structure

```
agent-memory-openenv/
├── openenv.yaml              # Environment spec
├── env/
│   ├── environment.py        # Gymnasium env (reset / step / render)
│   ├── state_builder.py      # Observation constructor
│   ├── action_handler.py     # Action dispatch
│   └── memory_manager.py     # Coordinates all memory layers
├── memory/
│   ├── working_memory.py     # FIFO buffer
│   ├── episodic_memory.py    # Timestamped events
│   └── semantic_memory.py    # Structured user model
├── tasks/
│   ├── easy_task.py          # Explicit fact recall
│   ├── medium_task.py        # Preference reasoning
│   └── hard_task.py          # Multi-memory reasoning
├── graders/
│   ├── storage_grader.py     # Memory allocation scoring
│   ├── retrieval_grader.py   # Retrieval relevance scoring
│   └── response_grader.py    # Response quality scoring
├── reward/
│   └── reward_function.py    # Weighted reward combiner
├── agents/
│   ├── baseline_agent.py     # Keyword heuristic
│   └── rl_agent.py           # PPO via stable-baselines3
├── evaluation/
│   ├── evaluate.py           # Run + score scenarios
│   └── metrics.py            # Aggregate metrics
├── data/conversations/
│   └── scenarios.py          # 9 built-in scenarios
├── scripts/
│   └── run_baseline.py       # CLI entry point
├── app/
│   └── app.py                # Gradio interactive demo
├── Dockerfile
├── requirements.txt
└── README.md
```

## Extending

**Add scenarios:** Edit `data/conversations/scenarios.py` — follow the existing dict schema with `ground_truth_storage`, `ground_truth_retrieval`, and keyword lists.

**Add memory importance scoring:** The episodic memory already has an `importance` field. Extend the storage grader to assign importance scores and the episodic memory to use them for smarter eviction.

**Plug in an LLM for response generation:** Replace the simple string concatenation in `environment.py` (the `response_text = ...` line in the query phase) with a call to a local model via `transformers`. The response grader already supports both keyword and semantic similarity modes.
