import numpy as np
from arm_benchmark.core.world import PandaWorld, HOME
from arm_benchmark.scenes.obstacles import generate_obstacles, save_obstacles, load_obstacles

w = PandaWorld(gui=False)
obs = generate_obstacles(w, n=10, seed=0)
print(f"requested 8, placed {len(obs)}")
for o in obs:
    print(f"  id={o.obstacle_id} {o.shape_type:8s} pos={np.round(o.position,2)} br={o.bounding_radius:.3f}")
print("home collision-free with obstacles:", w.is_collision_free(HOME))
print("min center-to-base dist:", round(min(np.linalg.norm(o.position) for o in obs), 3), "(must be > 0.25)")

save_obstacles(obs, "config/sparse_obstacles.json")
w.disconnect()

w2 = PandaWorld(gui=False)
obs2 = load_obstacles(w2, "config/sparse_obstacles.json")
print("reloaded count:", len(obs2), "| home collision-free after reload:", w2.is_collision_free(HOME))
w2.disconnect()
