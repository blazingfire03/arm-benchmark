import os
os.environ["NANOBIND_LEAK_WARNINGS"] = "0"

from arm_benchmark.core.world import PandaWorld
from arm_benchmark.scenes.obstacles import load_obstacles
from arm_benchmark.scenes.testcases import generate_astar_validated_cases, save_test_cases
from arm_benchmark.planners.astar import AStar

w = PandaWorld(gui=False)
load_obstacles(w, "config/sparse_obstacles.json")

print("generating A*-validated test cases (this may take a few minutes)...\n")
astar = AStar()
cases = generate_astar_validated_cases(w, astar, n=10, seed=42, max_candidates=50)
save_test_cases(cases, "config/sparse_test_cases.json")
w.disconnect()
print(f"\nsaved {len(cases)} cases to config/sparse_test_cases.json")
