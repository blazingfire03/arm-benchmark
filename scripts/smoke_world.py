import numpy as np
from arm_benchmark.core.world import PandaWorld, HOME
from arm_benchmark.core.types import Obstacle

w = PandaWorld(gui=False)
print("arm joint indices:", w.arm_joints)
print("flange link index:", w.flange_link)
print("ACM whitelisted self-pairs:", len(w.allowed_self_pairs))
print("home collision-free:", w.is_collision_free(HOME))
print("ee at home:", np.round(w.ee_position(HOME), 3))

# discrimination: random configs should be a MIX of self-clear and self-colliding
rng = np.random.default_rng(0)
n = 500
clear = sum(w.is_self_collision_free(rng.uniform(w.lower, w.upper)) for _ in range(n))
print(f"self-collision-free among {n} random configs: {clear}/{n}")

# obstacle check: a box straddling the home pose must register a collision
w.spawn(Obstacle(obstacle_id=0, shape_type="box",
                 position=np.array([0.3, 0.0, 0.5]),
                 dimensions=np.array([0.4, 0.4, 0.4])))
print("home collision-free after big box on arm:", w.is_collision_free(HOME))
w.disconnect()
