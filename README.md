---
title: SQL Query Optimizer Environment Server
emoji: 🐳
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# SQL Query Optimizer — OpenEnv Environment

An **OpenEnv-compliant** environment where AI agents learn to review, rewrite, and optimise SQL queries across three real-world failure patterns.

> **HF Spaces**: [param20h/sql-query-optimizer](https://huggingface.co/spaces/param20h/sql-query-optimizer)

---

## Environment Description

Real-world SQL anti-patterns cost companies millions in infrastructure. This environment teaches agents to identify and fix them through a reward-shaped episode loop. Each episode presents the agent with a broken or unoptimised query alongside schema context; the agent iteratively rewrites it until done or max steps are reached.

**Why this domain?**
- Used by data engineers and DBAs every day
- Deterministically gradeable (no ambiguous LLM judging)
- Natural difficulty progression from syntax errors to multi-factor optimisation

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | `int` | Task number (1–3) |
| `task_name` | `str` | Slug identifier |
| `task_description` | `str` | What the agent must accomplish |
| `query` | `str` | The SQL to fix |
| `schema_context` | `str` | Relevant DDL / table definitions |
| `hint` | `str \| null` | Optional hint (tasks 1 & 2 only) |
| `step_number` | `int` | Current step (0-indexed) |
| `max_steps` | `int` | Steps allowed per episode |
| `done` | `bool` | Whether episode has ended |

---

## Action Space

| Field | Type | Description |
|---|---|---|
| `rewritten_query` | `str` | The agent's improved SQL |
| `explanation` | `str` | Brief description of changes made |
| `is_done` | `bool` | `true` when the agent believes the query is fully fixed |

---

## Reward Design

The reward is **shaped** (not sparse) — the agent receives signal every step:

| Component | Value | Trigger |
|---|---|---|
| Delta reward | +0.0–0.50 × Δgrader | Grader score improves |
| Completion bonus | +0.50 | `is_done=True` and grader ≥ 0.80 |
| Partial completion | +grader × 0.30 | `is_done=True` (always) |
| Step penalty | −0.02 / step | After halfway point, if not done |
| Invalid penalty | −0.10 | Empty or unparseable query |

Final `score` per step is clamped to `[0.0, 1.0]`.

---

## Tasks

### Task 1 — `fix-broken-join` (Easy)
The query uses a comma-separated cross-join (`FROM orders, customers`) without any join condition, causing a Cartesian product. The agent must rewrite with `INNER JOIN … ON o.customer_id = c.customer_id`.

**Max steps**: 3 | **Grader**: checks JOIN keyword + ON clause with correct key

### Task 2 — `eliminate-n-plus-one` (Medium)
A correlated scalar subquery in the `SELECT` list executes once per row (N+1 problem). The agent must collapse it into a single `LEFT JOIN departments ON e.dept_id = d.dept_id`.

**Max steps**: 4 | **Grader**: checks subquery removal + JOIN on dept_id

### Task 3 — `full-optimization` (Hard)
Four independent issues to fix:
1. Remove redundant `DISTINCT` (PK join makes it unnecessary)
2. Replace `SELECT *` with explicit columns
3. Replace `CAST(price AS VARCHAR) LIKE '1%'` → `price >= 100 AND price < 200` (sargable)
4. Add an index hint comment for `(category, price)`

**Max steps**: 5 | **Grader**: 4 × 0.25 sub-criteria, fully independent

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/reset` | Start episode `{ "task_id": 1 }` |
| `POST` | `/step` | Submit action `{ "rewritten_query": "...", "explanation": "...", "is_done": true }` |
| `GET` | `/state` | Current internal state |
| `GET` | `/tasks` | All tasks + action schema |
| `GET` | `/grader` | Grader score for current episode |
| `POST` | `/baseline` | Run baseline inference (requires `OPENAI_API_KEY`) |

Interactive docs: `http://localhost:7860/docs`

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker
- `API_BASE_URL` (OpenAI-compatible endpoint for inference)
- `MODEL_NAME` (model identifier for inference)
- `HF_TOKEN` (API key / bearer token for inference)

### Local (Python)

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

### Local (Docker)

```bash
docker build -t sql-optimizer-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... sql-optimizer-env
```

### Baseline Inference

```bash
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="hf_or_openai_api_key_here"
python inference.py
```

### OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

### Deploy to HF Spaces

```bash
pip install huggingface_hub
huggingface-cli login
openenv push --repo-id your-username/sql-query-optimizer
```

### Environment Configuration

Define these variables before running inference or `/baseline`:

```powershell
$env:API_BASE_URL = "https://api.openai.com/v1"
$env:MODEL_NAME = "gpt-4o-mini"
$env:HF_TOKEN = "your_api_key"
```

---

## Baseline Scores

Measured with `gpt-4o-mini` at `temperature=0`, single-pass:

| Task | Name | Difficulty | Grader Score |
|---|---|---|---|
| 1 | fix-broken-join | Easy | 0.86 |
| 2 | eliminate-n-plus-one | Medium | 0.72 |
| 3 | full-optimization | Hard | 0.50 |
| — | **Average** | — | **0.69** |

> Scores are reproducible: same model, same temperature, same grader → same output.

---

## Project Structure

```
metaXscaler/
├── env/
│   ├── __init__.py
│   ├── environment.py   # reset(), step(), state()
│   ├── models.py        # Observation, Action, Reward (Pydantic)
│   ├── tasks.py         # Task definitions + graders
│   └── reward.py        # Shaped reward function
├── server.py            # FastAPI app
├── baseline.py          # Baseline inference script
├── openenv.yaml         # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## License

MIT
