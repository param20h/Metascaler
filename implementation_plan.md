# OpenEnv Round 1 — Implementation Plan

**Deadline: 7 April 2026, 11:59 PM IST**  
**Domain: SQL Query Optimization Review**

## Background

We need to build a fully compliant OpenEnv environment that an AI agent can interact with via `step()` / `reset()` / `state()`. The environment must model a real-world task, expose 3 graded tasks (easy → hard), include a shaped reward function, a baseline inference script, and deploy to Hugging Face Spaces via Docker.

**Chosen Domain: SQL Query Optimization Review**  
A data engineer/DBA reviews slow SQL queries and must rewrite or annotate them to be more performant and correct. This is a genuine, high-value task used in industry every day. It fills a gap in current OpenEnv submissions and is directly relevant to evaluating LLM coding/reasoning ability.

---

## Proposed Project Structure

```
metaXscaler/
├── env/
│   ├── __init__.py
│   ├── environment.py       # Core: reset(), step(), state()
│   ├── models.py            # Pydantic Observation, Action, Reward
│   ├── tasks.py             # Task definitions + graders
│   └── reward.py            # Shaped reward logic
├── server.py                # FastAPI server exposing all endpoints
├── baseline.py              # Baseline inference script (OpenAI API)
├── openenv.yaml             # OpenEnv metadata
├── Dockerfile               # Container build
├── requirements.txt
└── README.md
```

---

## User Review Required

> [!IMPORTANT]
> The chosen domain is **SQL Query Optimization Review**. If you prefer a different real-world domain (e.g. email triage, code review, scheduling), let me know before implementation begins.

> [!WARNING]
> The baseline script (`baseline.py`) requires a valid `OPENAI_API_KEY` environment variable at runtime. This is standard per spec but means actual inference costs money. The script will be structured to be minimal and reproducible.

---

## Proposed Changes

### Component 1 — Pydantic Models

#### [NEW] [models.py](file:///d:/metaXscaler/env/models.py)

Define three fully typed Pydantic models as required by OpenEnv spec:

- **`Observation`**: Contains `task_id`, `query` (the SQL to fix), `schema_context` (table definitions), `hint` (optional natural-language hint), `step_number`, `max_steps`.
- **`Action`**: Contains `rewritten_query` (str), `explanation` (str), `is_done` (bool).
- **`Reward`**: Contains `score` (float 0.0–1.0), `breakdown` (dict with partial scores), `feedback` (str).

---

### Component 2 — Core Environment

#### [NEW] [environment.py](file:///d:/metaXscaler/env/environment.py)

Implements the three required methods:

| Method | Behaviour |
|---|---|
| `reset(task_id)` | Loads the specified task, resets episode state, returns initial `Observation` |
| `step(action)` | Validates action, calls grader, computes reward, advances state, returns `(Observation, Reward, done, info)` |
| `state()` | Returns internal snapshot: current task, step count, cumulative score, history |

**Episode boundary**: `done=True` when `action.is_done=True` OR `step_number >= max_steps`.

---

### Component 3 — Tasks & Graders

#### [NEW] [tasks.py](file:///d:/metaXscaler/env/tasks.py)

Three tasks with deterministic programmatic graders:

| Task | Difficulty | Objective | Grader Logic |
|---|---|---|---|
| **Task 1** — Fix a broken `JOIN` | Easy | Rewrite a query with a missing `ON` clause or wrong join type | Checks syntactic correctness + presence of correct join key → score 0.0–1.0 |
| **Task 2** — Eliminate N+1 pattern | Medium | Collapse a correlated subquery into a single `JOIN` | Checks absence of subquery in `WHERE`, presence of `JOIN`, result equivalence via AST diff → score 0.0–1.0 |
| **Task 3** — Full optimisation | Hard | Remove redundant `DISTINCT`, add appropriate index hints, rewrite `SELECT *` to explicit columns, fix implicit type casts | Weighted rubric across 4 sub-criteria, each 0–0.25 → total 0.0–1.0 |

Graders are **deterministic**: same rewritten query → same score, every run.

---

### Component 4 — Reward Function

#### [NEW] [reward.py](file:///d:/metaXscaler/env/reward.py)

Shaped reward (not sparse):

- **Partial credit per step**: +0.0–0.5 for incremental improvement detected by grader
- **Completion bonus**: +0.5 if grader score ≥ 0.8 when `is_done=True`
- **Step penalty**: −0.02 per unnecessary step (> task's minimum required steps)
- **Invalid action penalty**: −0.1 for empty, unparseable, or destructive queries
- **Total episode reward**: sum of all step rewards, clamped to [0.0, 1.0]

---

### Component 5 — FastAPI Server & Endpoints

#### [NEW] [server.py](file:///d:/metaXscaler/server.py)

Exposes all required endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Body: `{"task_id": 1}` → returns initial `Observation` |
| `/step` | POST | Body: `Action` → returns `Observation`, `Reward`, `done`, `info` |
| `/state` | GET | Returns current internal state snapshot |
| `/tasks` | GET | Lists all tasks with descriptions + action schema |
| `/grader` | GET | Returns final grader score for the current completed episode |
| `/baseline` | POST | Triggers baseline inference on all 3 tasks, returns scores |

---

### Component 6 — Baseline Inference Script

#### [NEW] [baseline.py](file:///d:/metaXscaler/baseline.py)

- Uses `openai` Python client (not the deprecated library)
- Reads `OPENAI_API_KEY` from environment variable
- Loops over all 3 tasks: `reset()` → prompt LLM → `step()` → collect score
- Prints reproducible score table to stdout
- Self-contained — can be run as `python baseline.py`

---

### Component 7 — OpenEnv Metadata

#### [NEW] [openenv.yaml](file:///d:/metaXscaler/openenv.yaml)

```yaml
name: sql-query-optimizer
version: "1.0.0"
description: "AI agent reviews and rewrites SQL queries for correctness and performance."
tags: [openenv, sql, code-review, data-engineering]
tasks:
  - id: 1
    name: fix-broken-join
    difficulty: easy
  - id: 2
    name: eliminate-n-plus-one
    difficulty: medium
  - id: 3
    name: full-optimization
    difficulty: hard
```

---

### Component 8 — Dockerfile

#### [NEW] [Dockerfile](file:///d:/metaXscaler/Dockerfile)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

Port `7860` is the default HF Spaces port.

---

### Component 9 — README

#### [NEW] [README.md](file:///d:/metaXscaler/README.md)

Sections:
1. Environment description & motivation
2. Action space definition (with field types)
3. Observation space definition (with field types)
4. Task descriptions + expected difficulty
5. Setup & usage instructions (local + Docker + HF Spaces)
6. Baseline scores table

---

## Verification Plan

### Automated Tests

```bash
# 1. Validate OpenEnv spec compliance
openenv validate

# 2. Docker build + run
docker build -t sql-optimizer-env .
docker run -p 7860:7860 sql-optimizer-env

# 3. Smoke test all endpoints
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": 1}'
curl http://localhost:7860/tasks
curl http://localhost:7860/state
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"rewritten_query": "SELECT ...", "explanation": "...", "is_done": true}'
curl http://localhost:7860/grader

# 4. Run baseline script
OPENAI_API_KEY=sk-... python baseline.py
```

### Pre-Submission Checklist (all must pass)
- [ ] `openenv validate` → passes with no errors
- [ ] `docker build && docker run` → container starts cleanly on port 7860
- [ ] `/reset` returns 200 with valid `Observation` JSON
- [ ] `/tasks` returns list of 3 tasks
- [ ] `/grader` returns a score in `[0.0, 1.0]`
- [ ] `/baseline` returns 3 scores (one per task) without error
- [ ] HF Space URL responds to ping
- [ ] Grader scores are deterministic (run twice, same input → same score)

### HF Spaces Deployment
```bash
openenv push --repo-id <your-username>/sql-query-optimizer
```
After deploy, verify: `curl https://huggingface.co/spaces/<username>/sql-query-optimizer/reset`
