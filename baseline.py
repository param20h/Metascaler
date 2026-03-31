"""
Baseline inference script for the SQL Query Optimizer OpenEnv environment.

Usage:
    python baseline.py              # human-readable output
    python baseline.py --json       # JSON output (used by /baseline endpoint)

Requires:
    OPENAI_API_KEY environment variable

The script runs gpt-4o-mini against all 3 tasks and reports grader scores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from openai import OpenAI

# ── import env from local package ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from env.environment import SQLOptimizerEnv
from env.models import Action

# ──────────────────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"
MAX_STEPS = 5
TASKS = [1, 2, 3]

SYSTEM_PROMPT = """You are a database performance engineer.
You will receive a broken or unoptimised SQL query along with table schema context.
Your job is to rewrite the query so it is correct and performant.

Respond ONLY with a JSON object with these exact keys:
{
  "rewritten_query": "<your improved SQL>",
  "explanation": "<brief explanation of changes>",
  "is_done": true
}
Do not wrap in markdown. Output raw JSON only."""


def _build_user_message(obs_dict: dict) -> str:
    return (
        f"Task: {obs_dict['task_name']} ({obs_dict['task_id']} — difficulty: "
        f"{obs_dict.get('difficulty', 'unknown')})\n\n"
        f"Description:\n{obs_dict['task_description']}\n\n"
        f"Schema:\n{obs_dict['schema_context']}\n\n"
        f"Query to fix:\n{obs_dict['query']}"
        + (f"\n\nHint: {obs_dict['hint']}" if obs_dict.get("hint") else "")
    )


def run_baseline(verbose: bool = True) -> dict[str, float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    env = SQLOptimizerEnv()
    results: dict[str, float] = {}

    for task_id in TASKS:
        obs = env.reset(task_id=task_id)
        obs_dict = obs.model_dump()
        final_score = 0.0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Task {task_id}: {obs_dict['task_name']} [{obs_dict['task_id']}]")
            print(f"{'='*60}")

        for step_num in range(MAX_STEPS):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(obs_dict)},
            ]

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content.strip()
                parsed = json.loads(content)
                action = Action(
                    rewritten_query=parsed.get("rewritten_query", ""),
                    explanation=parsed.get("explanation", ""),
                    is_done=bool(parsed.get("is_done", False)),
                )
            except Exception as exc:
                if verbose:
                    print(f"  Step {step_num + 1}: LLM error — {exc}")
                action = Action(
                    rewritten_query="",
                    explanation="error",
                    is_done=True,
                )

            obs, reward, done, info = env.step(action)
            obs_dict = obs.model_dump()
            final_score = info["grader_score"]

            if verbose:
                print(
                    f"  Step {step_num + 1}: grader_score={info['grader_score']:.3f}  "
                    f"step_reward={reward.score:.4f}  feedback={reward.feedback[:80]}"
                )

            if done:
                break

        results[f"task_{task_id}_{env._task.name}"] = round(final_score, 4)

        if verbose:
            print(f"  → Final grader score: {final_score:.4f}")

    if verbose:
        print(f"\n{'='*60}")
        print("BASELINE RESULTS")
        print(f"{'='*60}")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        avg = sum(results.values()) / len(results)
        print(f"  Average: {avg:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv SQL Optimizer — Baseline Inference")
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON (used by /baseline endpoint)"
    )
    args = parser.parse_args()

    scores = run_baseline(verbose=not args.json)
    if args.json:
        print(json.dumps(scores))
