import time
import numpy as np
import pybullet as p
from .base import Planner
from .ompl_planners import OMPLPlanner
from ..core.types import Trajectory

N_WAYPOINTS = 50
N_ROLLOUTS = 10
MAX_ITER = 100
SMOOTH_WEIGHT = 1.0
OBS_WEIGHT = 500.0
OBS_EPSILON = 0.15
NOISE_STD = 0.10
TIME_BUDGET = 10.0


class STOMP(Planner):
    name = "STOMP"

    def __init__(self, n_waypoints=N_WAYPOINTS, n_rollouts=N_ROLLOUTS,
                 max_iter=MAX_ITER, smooth_w=SMOOTH_WEIGHT, obs_w=OBS_WEIGHT,
                 eps=OBS_EPSILON, noise_std=NOISE_STD, time_budget=TIME_BUDGET):
        self.n_wp = n_waypoints
        self.n_rollouts = n_rollouts
        self.max_iter = max_iter
        self.smooth_w = smooth_w
        self.obs_w = obs_w
        self.eps = eps
        self.noise_std = noise_std
        self.time_budget = time_budget
        self._seed_planner = OMPLPlanner("RRT-Connect", time_budget=2.0, simplify=True)

    def _smoothness_cost(self, traj):
        accel = traj[2:] - 2 * traj[1:-1] + traj[:-2]
        return float(np.sum(accel ** 2))

    def _obstacle_cost(self, world, traj):
        cost = 0.0
        for i in range(1, len(traj) - 1):
            world.set_config(traj[i])
            min_d = 1.0
            for ob in world.obstacles:
                pts = p.getClosestPoints(world.panda, ob.pybullet_body_id,
                                         distance=self.eps + 0.1, physicsClientId=world.cid)
                for cp in pts:
                    min_d = min(min_d, cp[8])
            if world.table is not None:
                pts = p.getClosestPoints(world.panda, world.table,
                                         distance=self.eps + 0.1, physicsClientId=world.cid)
                for cp in pts:
                    if cp[3] not in world.allowed_table_links:
                        min_d = min(min_d, cp[8])
            if min_d < self.eps:
                cost += (self.eps - min_d) ** 2
        return cost

    def _total_cost(self, world, traj):
        return (self.smooth_w * self._smoothness_cost(traj) +
                self.obs_w * self._obstacle_cost(world, traj))

    def _resample_to_n(self, waypoints, n):
        old_n = len(waypoints)
        if old_n == n:
            return waypoints.copy()
        old_t = np.linspace(0, 1, old_n)
        new_t = np.linspace(0, 1, n)
        resampled = np.zeros((n, waypoints.shape[1]))
        for j in range(waypoints.shape[1]):
            resampled[:, j] = np.interp(new_t, old_t, waypoints[:, j])
        return resampled

    def plan(self, world, start, goal) -> Trajectory:
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        n, dof = self.n_wp, 7
        t0 = time.perf_counter()
        rng = np.random.default_rng()

        seed = self._seed_planner.plan(world, start, goal)
        if seed.planning_success:
            traj = self._resample_to_n(seed.waypoints, n)
            seed_traj = traj.copy()
            init_method = "rrt_connect"
        else:
            traj = np.linspace(start, goal, n)
            seed_traj = traj.copy()
            init_method = "straight_line"

        traj[0] = start; traj[-1] = goal
        best_cost = self._total_cost(world, traj)
        best_traj = traj.copy()

        for it in range(self.max_iter):
            if time.perf_counter() - t0 > self.time_budget:
                break
            rollout_costs = []
            rollouts = []
            for _ in range(self.n_rollouts):
                noise = rng.normal(0, self.noise_std, (n, dof))
                noise[0] = 0; noise[-1] = 0
                candidate = np.clip(best_traj + noise, world.lower, world.upper)
                candidate[0] = start; candidate[-1] = goal
                # collision constraint: revert noisy waypoints that collide
                for i in range(1, n - 1):
                    if not world.is_collision_free(candidate[i]):
                        candidate[i] = seed_traj[i]
                c = self._total_cost(world, candidate)
                rollout_costs.append(c)
                rollouts.append(candidate)

            costs = np.array(rollout_costs)
            min_c = costs.min()
            exp_costs = np.exp(-10.0 * (costs - min_c) / (costs.max() - min_c + 1e-10))
            weights = exp_costs / exp_costs.sum()

            updated = np.zeros_like(traj)
            for w_i, rollout in zip(weights, rollouts):
                updated += w_i * rollout
            updated[0] = start; updated[-1] = goal

            # collision constraint on the weighted average too
            for i in range(1, n - 1):
                if not world.is_collision_free(updated[i]):
                    updated[i] = seed_traj[i]

            cost = self._total_cost(world, updated)
            if cost < best_cost:
                best_cost = cost
                best_traj = updated.copy()

        return Trajectory(waypoints=best_traj, dt=world.dt, planning_success=seed.planning_success,
                          metadata={"iterations": it + 1, "final_cost": float(best_cost),
                                    "init": init_method})
