import json
import numpy as np

MIN_EE_SEPARATION = 0.65


def _sample_valid_config(world, rng):
    while True:
        q = rng.uniform(world.lower, world.upper)
        if world.is_collision_free(q):
            return q


def generate_test_cases(world, n=50, seed=0, min_ee_sep=MIN_EE_SEPARATION):
    rng = np.random.default_rng(seed)
    cases = []
    cid = 0
    while len(cases) < n:
        start = _sample_valid_config(world, rng)
        goal = _sample_valid_config(world, rng)
        p_start, p_goal = world.ee_position(start), world.ee_position(goal)
        if np.linalg.norm(p_goal - p_start) < min_ee_sep:
            continue
        cases.append(dict(
            test_case_id=cid,
            start=start.tolist(),
            goal=goal.tolist(),
            start_ee=p_start.tolist(),
            goal_ee=p_goal.tolist(),
            straight_line_ee=float(np.linalg.norm(p_goal - p_start)),
            d_optimal=None,
        ))
        cid += 1
    return cases


def save_test_cases(cases, path):
    with open(path, "w") as f:
        json.dump(cases, f, indent=2)


def load_test_cases(path):
    with open(path) as f:
        return json.load(f)
