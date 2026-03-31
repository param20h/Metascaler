# OpenEnv Submission — Task Checklist

**Deadline: 7 April 2026, 11:59 PM IST**

## Phase 1: Design & Planning
- [ ] Choose real-world domain for the environment
- [ ] Define Observation, Action, Reward Pydantic models
- [ ] Design 3 tasks (easy → medium → hard) with graders
- [ ] Design reward function (partial progress, penalties)

## Phase 2: Core Environment Implementation
- [ ] Scaffold project with `openenv init`
- [ ] Implement `reset()` → returns clean initial observation
- [ ] Implement `step(action)` → returns observation, reward, done, info
- [ ] Implement `state()` → returns current internal state
- [ ] Create `openenv.yaml` with metadata
- [ ] Implement Task 1 (Easy) + grader (score 0.0–1.0)
- [ ] Implement Task 2 (Medium) + grader (score 0.0–1.0)
- [ ] Implement Task 3 (Hard) + grader (score 0.0–1.0)

## Phase 3: API Endpoints
- [ ] `POST /reset` — start/restart episode
- [ ] `POST /step` — advance environment
- [ ] `GET /state` — return current state
- [ ] `GET /tasks` — list tasks + action schemas
- [ ] `GET /grader` — return grader score after episode
- [ ] `POST /baseline` — trigger baseline inference, return scores

## Phase 4: Baseline Inference Script
- [ ] Write `baseline.py` using OpenAI API client
- [ ] Read `OPENAI_API_KEY` from environment variables
- [ ] Run model against all 3 tasks
- [ ] Output reproducible baseline scores

## Phase 5: Containerization & Deployment
- [ ] Write working `Dockerfile`
- [ ] Test `docker build && docker run` locally
- [ ] Deploy to Hugging Face Spaces (tagged `openenv`)
- [ ] Verify Space responds to `reset()` (200 OK)

## Phase 6: Validation & Documentation
- [ ] Run `openenv validate` — must pass
- [ ] Write `README.md` (description, action/obs spaces, task descriptions, setup, baseline scores)
- [ ] Final pre-submission checklist pass
- [ ] Submit HF Spaces URL before deadline
