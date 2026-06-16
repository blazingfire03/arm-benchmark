import numpy as np
from arm_benchmark.core.world import PandaWorld
from arm_benchmark.scenes.obstacles import generate_obstacles, save_obstacles
from arm_benchmark.scenes.testcases import generate_test_cases
from arm_benchmark.planners.min_jerk import MinJerk
from arm_benchmark.planners.utils import retime_minjerk
from arm_benchmark.core.types import Trajectory

N_TRIALS = 30   # quick calibration trials per count
COUNTS = [2, 4, 6, 8, 12, 16, 20, 25, 30, 40, 50]

mj = MinJerk()

print(f"OBSTACLE CALIBRATION — min-jerk collision rate vs obstacle count")
print(f"  {N_TRIALS} trials per count\n")
print(f"{'count':>6} {'collision %':>12} {'placed':>8} {'target':>20}")
print("-" * 55)

for n_obs in COUNTS:
    w = PandaWorld(gui=False, acm_samples=200)
    obs = generate_obstacles(w, n_obs, seed=0)
    actual = len(obs)
    cases = generate_test_cases(w, n=N_TRIALS, seed=100)

    collisions = 0
    for case in cases:
        start, goal = np.array(case["start"]), np.array(case["goal"])
        traj = mj.plan(w, start, goal)
        retimed = retime_minjerk(traj.waypoints, dt=w.dt, max_speed=1.0, min_duration=0.5)
        for q in retimed:
            if w.contact_count(q) > 0:
                collisions += 1
                break

    rate = collisions / N_TRIALS * 100
    if rate <= 20:
        target = "<= 20% (SPARSE)"
    elif rate <= 70:
        target = "50-70% (MODERATE)"
    elif rate >= 95:
        target = ">= 95% (DENSE)"
    else:
        target = ""
    print(f"{n_obs:>6} {rate:>10.1f}% {actual:>8} {target:>20}")

    w.disconnect()

print(f"\nPick counts that hit: sparse <=20%, moderate 50-70%, dense >=95%")
