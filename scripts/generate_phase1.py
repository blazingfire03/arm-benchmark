import os
from arm_benchmark.core.world import PandaWorld
from arm_benchmark.scenes.obstacles import generate_obstacles, save_obstacles
from arm_benchmark.scenes.testcases import generate_test_cases, save_test_cases

os.makedirs("config", exist_ok=True)

SUB_PHASES = {
    "sparse":   {"n_obs": 4,  "seed": 0},
    "moderate": {"n_obs": 12, "seed": 1},
    "dense":    {"n_obs": 16, "seed": 2},
}

for name, cfg in SUB_PHASES.items():
    print(f"\n--- {name.upper()} (n={cfg['n_obs']}) ---")
    w = PandaWorld(gui=False, acm_samples=200)
    obs = generate_obstacles(w, cfg["n_obs"], seed=cfg["seed"])
    print(f"  placed {len(obs)} obstacles")
    save_obstacles(obs, f"config/{name}_obstacles.json")

    cases = generate_test_cases(w, n=50, seed=cfg["seed"] + 100)
    print(f"  generated {len(cases)} test cases")
    save_test_cases(cases, f"config/{name}_test_cases.json")
    w.disconnect()

print("\nsaved:")
for name in SUB_PHASES:
    print(f"  config/{name}_obstacles.json")
    print(f"  config/{name}_test_cases.json")
print("\nready for Phase 1 run")
