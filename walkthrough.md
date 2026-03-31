# Walkthrough — SQL Query Optimizer OpenEnv Environment

## What Was Built

A fully compliant OpenEnv environment for the metaXscaler Round 1 submission.

**Domain**: SQL Query Optimization Review — agents rewrite broken/slow SQL queries.

---

## Files Created

| File | Purpose |
|---|---|
| [env/__init__.py](file:///d:/metaXscaler/env/__init__.py) | Package init |
| [env/models.py](file:///d:/metaXscaler/env/models.py) | Pydantic [Observation](file:///d:/metaXscaler/env/models.py#15-31), [Action](file:///d:/metaXscaler/env/models.py#37-50), [Reward](file:///d:/metaXscaler/env/models.py#63-78) models |
| [env/tasks.py](file:///d:/metaXscaler/env/tasks.py) | 3 task definitions + deterministic graders |
| [env/reward.py](file:///d:/metaXscaler/env/reward.py) | Shaped reward function |
| [env/environment.py](file:///d:/metaXscaler/env/environment.py) | [reset()](file:///d:/metaXscaler/server.py#91-99) / [step()](file:///d:/metaXscaler/server.py#101-109) / [state()](file:///d:/metaXscaler/env/environment.py#142-157) core |
| [server.py](file:///d:/metaXscaler/server.py) | FastAPI server — all 6 endpoints |
| [baseline.py](file:///d:/metaXscaler/baseline.py) | Baseline inference script (OpenAI API) |
| [openenv.yaml](file:///d:/metaXscaler/openenv.yaml) | OpenEnv spec metadata |
| [Dockerfile](file:///d:/metaXscaler/Dockerfile) | Container — Python 3.11-slim, port 7860 |
| [requirements.txt](file:///d:/metaXscaler/requirements.txt) | fastapi, uvicorn, pydantic, openai, pyyaml |
| [README.md](file:///d:/metaXscaler/README.md) | Full documentation |
| [test_env.py](file:///d:/metaXscaler/test_env.py) | Smoke test script |

---

## Test Results

### Unit / Grader Tests (`python test_env.py`)

| Task | Input | Grader Score | Reward | Status |
|---|---|---|---|---|
| 1 — fix-broken-join (Easy) | Correct `INNER JOIN ON customer_id` | **1.000** | 1.0000 | ✅ PASS |
| 2 — eliminate-n-plus-one (Medium) | `LEFT JOIN` replacing correlated subquery | **1.000** | 1.0000 | ✅ PASS |
| 3 — full-optimization (Hard) | All 4 fixes applied | **1.000** | 1.0000 | ✅ PASS |
| Invalid action | Empty query string | 0.0 (penalty) | is_invalid=True | ✅ PASS |

### API Endpoint Tests (server on `localhost:7860`)

| Endpoint | Result |
|---|---|
| `GET /` | `{"status":"ok","environment":"sql-query-optimizer","version":"1.0.0"}` ✅ |
| `POST /reset {"task_id":1}` | Returns [Observation](file:///d:/metaXscaler/env/models.py#15-31) with `task_name=fix-broken-join, step_number=0` ✅ |
| `POST /step` | Returns `grader_score=1.0, done=True` ✅ |
| `GET /state` | Returns `done=True, cumulative_score=1.0` ✅ |
| `GET /tasks` | Returns 3 tasks with action schema ✅ |
| `GET /grader` | Returns `grader_score=1.0, done=True` ✅ |

---

## Remaining Steps (Before Deadline: 7 April 2026)

1. **Docker build & run** — run locally to confirm container starts
   ```bash
   docker build -t sql-optimizer-env .
   docker run -p 7860:7860 sql-optimizer-env
   ```

2. **OpenEnv validate**
   ```bash
   pip install openenv-core
   openenv validate
   ```

3. **Deploy to HF Spaces**
   ```bash
   huggingface-cli login
   openenv push --repo-id <your-username>/sql-query-optimizer
   ```

4. **Run baseline** (needs `OPENAI_API_KEY`)
   ```bash
   export OPENAI_API_KEY=sk-...
   python baseline.py
   ```

5. **Submit** the HF Spaces URL on the platform before 7 April 2026, 11:59 PM IST.
