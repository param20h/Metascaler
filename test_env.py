"""Quick smoke test for all 3 tasks."""
import sys, json
sys.path.insert(0, ".")

from env.environment import SQLOptimizerEnv
from env.models import Action

env = SQLOptimizerEnv()

# ── Task 1 ──────────────────────────────────────────────────────────────────
print("=== Task 1 (Easy): fix-broken-join ===")
obs = env.reset(1)
print(f"  task: {obs.task_name}")
action = Action(
    rewritten_query=(
        "SELECT o.order_id, c.name, o.total "
        "FROM orders o INNER JOIN customers c ON o.customer_id = c.customer_id "
        "WHERE o.total > 100"
    ),
    explanation="Replaced comma cross-join with INNER JOIN ON customer_id",
    is_done=True,
)
obs2, reward, done, info = env.step(action)
print(f"  grader_score={info['grader_score']:.3f}  step_reward={reward.score:.4f}  done={done}")
print(f"  feedback: {reward.feedback}")
assert obs2.done == True, "done should be True"
assert info["grader_score"] >= 0.8, f"Expected >=0.8, got {info['grader_score']}"

# ── Task 2 ──────────────────────────────────────────────────────────────────
print()
print("=== Task 2 (Medium): eliminate-n-plus-one ===")
obs = env.reset(2)
print(f"  task: {obs.task_name}")
action = Action(
    rewritten_query=(
        "SELECT e.name, d.dept_name "
        "FROM employees e "
        "LEFT JOIN departments d ON e.dept_id = d.dept_id "
        "WHERE e.salary > 50000"
    ),
    explanation="Replaced correlated subquery with a single LEFT JOIN",
    is_done=True,
)
obs2, reward, done, info = env.step(action)
print(f"  grader_score={info['grader_score']:.3f}  step_reward={reward.score:.4f}  done={done}")
print(f"  feedback: {reward.feedback}")
assert info["grader_score"] >= 0.7, f"Expected >=0.7, got {info['grader_score']}"

# ── Task 3 ──────────────────────────────────────────────────────────────────
print()
print("=== Task 3 (Hard): full-optimization ===")
obs = env.reset(3)
print(f"  task: {obs.task_name}")
action = Action(
    rewritten_query=(
        "-- Index hint: consider CREATE INDEX ON products(category, price)\n"
        "SELECT p.name, p.category, p.price, oi.quantity, oi.unit_price\n"
        "FROM   products p\n"
        "JOIN   order_items oi ON p.product_id = oi.product_id\n"
        "WHERE  p.price >= 100 AND p.price < 200\n"
        "  AND  p.category = 'Electronics'\n"
        "ORDER  BY p.name"
    ),
    explanation="Removed DISTINCT and SELECT *, replaced CAST LIKE with range, added index hint",
    is_done=True,
)
obs2, reward, done, info = env.step(action)
print(f"  grader_score={info['grader_score']:.3f}  step_reward={reward.score:.4f}  done={done}")
print(f"  feedback: {reward.feedback}")
assert info["grader_score"] >= 0.9, f"Expected >=0.9, got {info['grader_score']}"

# ── state() ─────────────────────────────────────────────────────────────────
print()
print("=== state() ===")
print(json.dumps(env.state(), indent=2))

# ── invalid action penalty ───────────────────────────────────────────────────
print()
print("=== Invalid action test ===")
env.reset(1)
obs2, reward, done, info = env.step(Action(rewritten_query="", explanation="", is_done=False))
print(f"  step_reward={reward.score}  is_invalid={info['is_invalid']}")
assert info["is_invalid"] == True, "Empty query should be flagged invalid"

print()
print("ALL TESTS PASSED")
