import json
import numpy as np
import pybullet as p
from ..core.types import Obstacle

# PLACEHOLDERS (confirm with Swapnil): obstacles around the arm, some on the table, some in air
R_MIN, R_MAX = 0.12, 0.58      # radial ring around the arm base (m)
REACH = 0.80                   # keep obstacles inside the arm's reachable sphere
AIR_FRACTION = 0.55             # fraction floating above the table
Z_AIR = (0.10, 0.70)           # floating height above the table top
MIN_OBSTACLE_GAP = 0.01
MAX_ATTEMPTS = 300

SHAPES = ("box", "cylinder", "sphere")
DIM_RANGES = dict(
    box=dict(half=(0.08, 0.15)),
    cylinder=dict(radius=(0.06, 0.10), height=(0.20, 0.40)),
    sphere=dict(radius=(0.08, 0.13)),
)


def _sample_shape(rng):
    shape = str(rng.choice(SHAPES))
    if shape == "box":
        dims = rng.uniform(*DIM_RANGES["box"]["half"], size=3)
        return shape, dims, float(np.linalg.norm(dims)), float(dims[2])
    if shape == "cylinder":
        r = rng.uniform(*DIM_RANGES["cylinder"]["radius"])
        h = rng.uniform(*DIM_RANGES["cylinder"]["height"])
        return shape, np.array([r, h]), float(np.hypot(r, h / 2)), h / 2
    r = rng.uniform(*DIM_RANGES["sphere"]["radius"])
    return shape, np.array([r]), float(r), r


def generate_obstacles(world, n, seed=0):
    rng = np.random.default_rng(seed)
    base = world.base_pos
    accepted = []
    world.reset_home()
    for oid in range(n):
        placed = False
        for _ in range(MAX_ATTEMPTS):
            shape, dims, br, half_h = _sample_shape(rng)
            r, th = rng.uniform(R_MIN, R_MAX), rng.uniform(0, 2 * np.pi)
            if rng.random() < AIR_FRACTION:
                z = base[2] + rng.uniform(*Z_AIR)          # floating
            else:
                z = base[2] + half_h                       # resting on table
            pos = np.array([base[0] + r * np.cos(th), base[1] + r * np.sin(th), z])
            if np.linalg.norm(pos - base) > REACH:
                continue
            if any(np.linalg.norm(pos - o.position) < br + o.bounding_radius + MIN_OBSTACLE_GAP
                   for o in accepted):
                continue
            ob = Obstacle(obstacle_id=oid, shape_type=shape, position=pos,
                          dimensions=dims, bounding_radius=br)
            world.spawn(ob)
            world.reset_home()
            p.performCollisionDetection(physicsClientId=world.cid)
            if p.getContactPoints(bodyA=world.panda, bodyB=ob.pybullet_body_id,
                                  physicsClientId=world.cid):
                p.removeBody(ob.pybullet_body_id, physicsClientId=world.cid)
                world.obstacles.remove(ob)
                continue
            accepted.append(ob)
            placed = True
            break
        if not placed:
            break
    return accepted


def save_obstacles(obstacles, path):
    data = [dict(obstacle_id=o.obstacle_id, shape_type=o.shape_type,
                 position=o.position.tolist(), dimensions=o.dimensions.tolist(),
                 orientation=o.orientation.tolist(), bounding_radius=o.bounding_radius)
            for o in obstacles]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_obstacles(world, path):
    with open(path) as f:
        data = json.load(f)
    obstacles = []
    for d in data:
        ob = Obstacle(obstacle_id=d["obstacle_id"], shape_type=d["shape_type"],
                      position=np.array(d["position"]), dimensions=np.array(d["dimensions"]),
                      orientation=np.array(d["orientation"]),
                      bounding_radius=d["bounding_radius"])
        world.spawn(ob)
        obstacles.append(ob)
    return obstacles
