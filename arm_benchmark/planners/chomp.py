import time
import numpy as np
import pybullet as p
from .base import Planner
from .ompl_planners import OMPLPlanner
from ..core.types import Trajectory

N_WAYPOINTS = 50
MAX_ITER = 300
LEARNING_RATE = 0.15
SMOOTH_WEIGHT = 0.5
OBS_WEIGHT = 800.0
OBS_EPSILON = 0.15
TIME_BUDGET = 10.0


class CHOMP(Planner):
    name = "CHOMP"

    def __init__(self, n_waypoints=N_WAYPOINTS, max_iter=MAX_ITER, lr=LEARNING_RATE,
                 smooth_w=SMOOTH_WEIGHT, obs_w=OBS_WEIGHT, eps=OBS_EPSILON,
                 time_budget=TIME_BUDGET):
        self.n_wp = n_waypoints
        self.max_iter = max_iter
        self.lr = lr
        self.smooth_w = smooth_w
        self.obs_w = obs_w
        self.eps = eps
        self.time_budget = time_budget
        self._seed_planner = OMPLPlanner("RRT-Connect", time_budget=2.0, simplify=True)

    def _build_smoothness_matrix(self, n):
        K = np.zeros((n - 2, n))
        for i in range(n - 2):
            K[i, i] = 1; K[i, i + 1] = -2; K[i, i + 2] = 1
        return K.T @ K

    def _signed_min_dist(self, world, q):
        world.set_config(q)
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
        return min_d

    def _obstacle_cost(self, d):
        if d >= self.eps:
            return 0.0
        return (self.eps - d) ** 2

    def _obstacle_cost_and_grad(self, world, traj):
        n, dof = traj.shape
        cost = 0.0
        grad = np.zeros_like(traj)
        delta = 0.003
        for i in range(1, n - 1):
            d = self._signed_min_dist(world, traj[i])
            c = self._obstacle_cost(d)
            if c < 1e-12:
                continue
            cost += c
            for j in range(dof):
                dq = np.zeros(dof); dq[j] = delta
                d_p = self._signed_min_dist(world, traj[i] + dq)
                d_m = self._signed_min_dist(world, traj[i] - dq)
                grad[i, j] = (self._obstacle_cost(d_p) - self._obstacle_cost(d_m)) / (2 * delta)
        return cost, grad

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
        n = self.n_wp
        t0 = time.perf_counter()

        seed = self._seed_planner.plan(world, start, goal)
        if seed.planning_success:
            traj = self._resample_to_n(seed.waypoints, n)
            seed_traj = traj.copy()  # keep for collision revert
            init_method = "rrt_connect"
        else:
            traj = np.linspace(start, goal, n)
            seed_traj = traj.copy()
            init_method = "straight_line"

        traj[0] = start; traj[-1] = goal
        A = self._build_smoothness_matrix(n)
        A_inner = A[1:n-1, 1:n-1]
        A_inv = np.linalg.pinv(A_inner + 1e-4 * np.eye(n - 2))

        best_traj = traj.copy()
        best_cost = float('inf')
        prev_cost = float('inf')
        stall_count = 0

        for it in range(self.max_iter):
            if time.perf_counter() - t0 > self.time_budget:
                break
            smooth_grad = (A @ traj)[1:n-1]
            obs_cost, obs_grad_full = self._obstacle_cost_and_grad(world, traj)
            obs_grad = obs_grad_full[1:n-1]
            total_grad = self.smooth_w * smooth_grad + self.obs_w * obs_grad
            update = A_inv @ total_grad
            traj[1:n-1] -= self.lr * update
            traj[1:n-1] = np.clip(traj[1:n-1], world.lower, world.upper)
            traj[0] = start; traj[-1] = goal

            # collision constraint: revert waypoints that moved into collision
            for i in range(1, n - 1):
                if not world.is_collision_free(traj[i]):
                    traj[i] = seed_traj[i]

            total_cost = self.smooth_w * np.sum(smooth_grad**2) + self.obs_w * obs_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_traj = traj.copy()
            if abs(prev_cost - total_cost) < 1e-6 * max(abs(prev_cost), 1):
                stall_count += 1
                if stall_count > 5:
                    break
            else:
                stall_count = 0
            prev_cost = total_cost

        return Trajectory(waypoints=best_traj, dt=world.dt, planning_success=seed.planning_success,
                          metadata={"iterations": it + 1, "final_cost": float(best_cost),
                                    "init": init_method})
