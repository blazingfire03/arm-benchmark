import numpy as np
from arm_benchmark.core.world import PandaWorld
from arm_benchmark.scenes.obstacles import load_obstacles
from arm_benchmark.scenes.testcases import generate_test_cases, save_test_cases, load_test_cases

w = PandaWorld(gui=False)
load_obstacles(w, "config/sparse_obstacles.json")
cases = generate_test_cases(w, n=50, seed=0)
print("generated cases:", len(cases))

seps = [c["straight_line_ee"] for c in cases]
print("EE separation  min %.3f  mean %.3f  max %.3f" % (min(seps), np.mean(seps), max(seps)))

ok = all(w.is_collision_free(np.array(c["start"])) and w.is_collision_free(np.array(c["goal"]))
         for c in cases)
print("all start/goal collision-free:", ok)

starts = {tuple(np.round(c["start"], 4)) for c in cases}
print("unique starts:", len(starts), "of", len(cases))

save_test_cases(cases, "config/sparse_test_cases.json")
reloaded = load_test_cases("config/sparse_test_cases.json")
print("reloaded:", len(reloaded), "| d_optimal placeholder:", reloaded[0]["d_optimal"])
w.disconnect()
