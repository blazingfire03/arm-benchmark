import numpy as np
from .base import Planner
from ..core.types import Trajectory

# PLACEHOLDERS — tune with Swapnil
K_ATT = 5.0           # attractive gain
K_REP = 0.8           # repulsive gain
D0 = 0.15             # obstacle influence radius (m)
STEP_SIZE = 0.02      # joint-space step (rad)
GOAL_TOL = 0.05       # goal-reached tolerance (rad, per joint max)
TIME_BUDGET = 5.0     # seconds
MAX_STEPS = 10000


class APF(Planner):
    name = "APF"

    def __init__(self, k_att=K_ATT, k_rep=K_REP, d0=D0, step_size=STEP_SIZE,
                 goal_tol=GOAL_TOL, time_budget=TIME_BUDGET, max_steps=MAX_STEPS):
        self.k_att = k_att
        self.k_rep = k_rep
        self.d0 = d0
        self.step_size = step_size
        self.goal_tol = goal_tol
        self.time_budget = time_budget
        self.max_steps = max_steps

    def _attractive_force(self, q, goal):
        """Pull toward goal in joint space."""
        diff = goal - q
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return np.zeros(7)
        return self.k_att * diff / dist

    def _repulsive_force(self, world, q):
        """Push away from nearby obstacles using EE-to-obstacle distance."""
        import pybullet as p
        world.set_config(q)
        force = np.zeros(7)
        ee_pos = np.array(p.getLinkState(world.panda, world.flange_link,
                                         physicsClientId=world.cid)[4])
        for ob in world.obstacles:
            closest = p.getClosestPoints(world.panda, ob.pybullet_body_id,
                                         distance=self.d0, physicsClientId=world.cid)
            if not closest:
                continue
            for cp in closest:
                dist = max(cp[8], 1e-6)  # contact distance (negative = penetrating)
                if dist >= self.d0:
                    continue
                # point on arm closest to obstacle
                pt_arm = np.array(cp[5])
                pt_obs = np.array(cp[6])
                direction = pt_arm - pt_obs
                norm = np.linalg.norm(direction)
                if norm < 1e-9:
                    continue
                direction /= norm
                # repulsive magnitude: grows as distance shrinks
                mag = self.k_rep * (1.0 / dist - 1.0 / self.d0) / (dist * dist)
                # project into joint space via numerical Jacobian approximation
                # (perturb each joint, measure EE shift toward obstacle)
                for j in range(7):
                    dq = np.zeros(7)
                    dq[j] = 0.001
                    world.set_config(q + dq)
                    ee_shift = np.array(p.getLinkState(world.panda, world.flange_link,
                                                       physicsClientId=world.cid)[4]) - ee_pos
                    force[j] += mag * np.dot(ee_shift, direction)
        return force

    def plan(self, world, start, goal) -> Trajectory:
        import time as _time
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        q = start.copy()
        path = [q.copy()]
        t0 = _time.perf_counter()

        for step in range(self.max_steps):
            if _time.perf_counter() - t0 > self.time_budget:
                return Trajectory(waypoints=np.array(path), dt=world.dt,
                                  planning_success=False,
                                  metadata={"reason": "timeout", "steps": step})

            if np.max(np.abs(q - goal)) < self.goal_tol:
                path.append(goal.copy())
                return Trajectory(waypoints=np.array(path), dt=world.dt,
                                  planning_success=True,
                                  metadata={"steps": step})

            f_att = self._attractive_force(q, goal)
            f_rep = self._repulsive_force(world, q)
            f_total = f_att + f_rep
            norm = np.linalg.norm(f_total)
            if norm < 1e-9:
                # stuck in local minimum — add random perturbation
                f_total = np.random.randn(7) * 0.1
                norm = np.linalg.norm(f_total)

            step_vec = self.step_size * f_total / norm
            q_new = np.clip(q + step_vec, world.lower, world.upper)

            q = q_new
            path.append(q.copy())

        return Trajectory(waypoints=np.array(path), dt=world.dt,
                          planning_success=False,
                          metadata={"reason": "max_steps", "steps": self.max_steps})
