import os, json, time
os.environ["NANOBIND_LEAK_WARNINGS"] = "0"

from arm_benchmark.core.world import PandaWorld
from arm_benchmark.scenes.obstacles import load_obstacles
from arm_benchmark.scenes.testcases import save_test_cases
from arm_benchmark.planners.astar import AStar
import numpy as np

SUB_PHASES = {
    "sparse":   {"obs": "config/sparse_obstacles.json",   "n": 50, "seed_start": 100},
    "moderate": {"obs": "config/moderate_obstacles.json", "n": 50, "seed_start": 200},
    "dense":    {"obs": "config/dense_obstacles.json",    "n": 50, "seed_start": 300},
}

MIN_EE_SEP = 0.65
astar = AStar()

for sp, cfg in SUB_PHASES.items():
    print(f"\n{'='*60}")
    print(f"  {sp.upper()} — generating A*-validated test cases")
    print(f"{'='*60}")

    w = PandaWorld(gui=False, acm_samples=200)
    load_obstacles(w, cfg["obs"])

    cases = []
    seed = cfg["seed_start"]
    attempts = 0
    max_attempts = 500

    while len(cases) < cfg["n"] and attempts < max_attempts:
        rng = np.random.default_rng(seed)
        seed += 1
        attempts += 1

        # sample collision-free start and goal
        start, goal = None, None
        for _ in range(100):
            q = rng.uniform(w.lower, w.upper)
            if w.is_collision_free(q):
                if start is None:
                    start = q
                else:
                    goal = q
                    break
        if start is None or goal is None:
            continue

        # check EE separation
        p_start = w.ee_position(start)
        p_goal = w.ee_position(goal)
        if np.linalg.norm(p_goal - p_start) < MIN_EE_SEP:
            continue

        # run A*
        t0 = time.perf_counter()
        traj = astar.plan(w, start, goal)
        wall = time.perf_counter() - t0

        if not traj.planning_success:
            reason = traj.metadata.get("reason", "?")
            print(f"  attempt {attempts:>3}: SKIP ({wall:.1f}s) — {reason[:50]}")
            continue

        d_opt = traj.metadata.get("d_optimal")
        if d_opt is None or d_opt <= 0:
            print(f"  attempt {attempts:>3}: SKIP — no valid d_optimal")
            continue

        cid = len(cases)
        cases.append(dict(
            test_case_id=cid,
            start=start.tolist(),
            goal=goal.tolist(),
            start_ee=p_start.tolist(),
            goal_ee=p_goal.tolist(),
            straight_line_ee=float(np.linalg.norm(p_goal - p_start)),
            d_optimal=d_opt,
            astar_planning_time=wall,
            astar_method=traj.metadata.get("method", "?"),
            astar_bins=traj.metadata.get("bins", "?"),
        ))
        print(f"  attempt {attempts:>3}: ACCEPTED case {cid} ({wall:.1f}s, d_opt={d_opt:.3f}m, {traj.metadata.get('method','?')})")

    w.disconnect()

    out = f"config/{sp}_test_cases.json"
    save_test_cases(cases, out)
    print(f"\n  saved {len(cases)}/{cfg['n']} cases to {out}")
    if len(cases) < cfg["n"]:
        print(f"  WARNING: only {len(cases)} cases found in {attempts} attempts")
