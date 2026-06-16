import time
import heapq
import itertools
import numpy as np
from .base import Planner
from .utils import shortcut, densify, segment_collision_free
from ..core.types import Trajectory

BINS = 16
TIME_BUDGET = 30.0
MAX_EXPANSIONS = 2_000_000


class AStar(Planner):
    name = "a_star"

    def __init__(self, bins=BINS, time_budget=TIME_BUDGET, max_expansions=MAX_EXPANSIONS,
                 shortcut_step=0.02):
        self.bins = bins
        self.time_budget = time_budget
        self.max_expansions = max_expansions
        self.shortcut_step = shortcut_step

    def _result(self, world, corner_path, t0, meta):
        dense = densify(corner_path, self.shortcut_step)
        # A* contract: collision-free or fail
        for q in dense:
            if not world.is_collision_free(q):
                return Trajectory(waypoints=np.array([corner_path[0], corner_path[-1]]),
                                  dt=world.dt, planning_success=False,
                                  metadata={"reason": "post-shortcut collision", "d_optimal": None, **meta})
        ee_pts = np.array([world.ee_position(q) for q in dense])
        d_opt = float(np.sum(np.linalg.norm(np.diff(ee_pts, axis=0), axis=1)))
        # validate: A* must return collision-free or fail
        for q in dense:
            if not world.is_collision_free(q):
                return Trajectory(waypoints=np.array([np.asarray(corner_path[0]), np.asarray(corner_path[-1])]),
                                  dt=world.dt, planning_success=False,
                                  metadata={"reason": "post-shortcut collision", "d_optimal": None, **meta})
        meta = {"d_optimal": d_opt, "internal_plan_time": time.perf_counter() - t0, **meta}
        return Trajectory(waypoints=dense, dt=world.dt, planning_success=True, metadata=meta)

    def _find_free_cell(self, world, q, lower, bw, bins):
        """Snap q to the nearest free grid cell. Try the natural cell first,
        then expand to 3^7 neighbors (±1 bin each joint)."""
        base = tuple(np.clip(np.floor((np.asarray(q) - lower) / bw).astype(int), 0, bins - 1))
        config_of = lambda idx: lower + (np.array(idx) + 0.5) * bw
        if world.is_collision_free(config_of(base)):
            return base
        # search neighboring cells sorted by joint-space distance to q
        candidates = []
        for offsets in itertools.product([-1, 0, 1], repeat=7):
            nb = tuple(np.clip(np.array(base) + np.array(offsets), 0, bins - 1))
            if nb == base:
                continue
            c = config_of(nb)
            candidates.append((float(np.linalg.norm(c - q)), nb))
        candidates.sort()
        for _, nb in candidates:
            if world.is_collision_free(config_of(nb)):
                return nb
        return None

    def plan(self, world, start, goal) -> Trajectory:
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        t0 = time.perf_counter()

        if segment_collision_free(world, start, goal, self.shortcut_step):
            return self._result(world, [start, goal], t0,
                                {"method": "direct", "cells_expanded": 0,
                                 "grid_cells": 0, "shortcut_corners": 2})

        lower, upper, bins = world.lower, world.upper, self.bins
        bw = (upper - lower) / bins

        def config_of(idx):
            return lower + (np.array(idx) + 0.5) * bw

        ee_cache, free_cache = {}, {}

        def ee(idx):
            if idx not in ee_cache:
                ee_cache[idx] = world.ee_position(config_of(idx))
            return ee_cache[idx]

        def free(idx):
            if idx not in free_cache:
                free_cache[idx] = world.is_collision_free(config_of(idx))
            return free_cache[idx]

        def fail(reason, n):
            return Trajectory(waypoints=np.array([start, goal]), dt=world.dt,
                              planning_success=False,
                              metadata={"reason": reason, "d_optimal": None, "cells_expanded": n})

        start_idx = self._find_free_cell(world, start, lower, bw, bins)
        goal_idx = self._find_free_cell(world, goal, lower, bw, bins)
        if start_idx is None or goal_idx is None:
            return fail("no free cell near start/goal", 0)

        goal_ee = ee(goal_idx)

        def h(idx):
            return float(np.linalg.norm(ee(idx) - goal_ee))

        counter = itertools.count()
        open_heap = [(h(start_idx), next(counter), start_idx)]
        g = {start_idx: 0.0}
        came, closed = {}, set()
        expansions = 0

        while open_heap:
            if time.perf_counter() - t0 > self.time_budget or expansions > self.max_expansions:
                return fail("timeout/expansion cap", expansions)
            _, _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal_idx:
                grid = [cur]
                while cur in came:
                    cur = came[cur]; grid.append(cur)
                grid.reverse()
                corners = [start] + [config_of(i) for i in grid] + [goal]
                sc = shortcut(world, corners, self.shortcut_step)
                return self._result(world, sc, t0,
                                    {"method": "grid", "cells_expanded": expansions,
                                     "grid_cells": len(grid), "shortcut_corners": len(sc)})
            closed.add(cur)
            expansions += 1
            cur_ee, cur_g = ee(cur), g[cur]
            for j in range(7):
                for step in (-1, 1):
                    nb = list(cur); nb[j] += step
                    if nb[j] < 0 or nb[j] >= bins:
                        continue
                    nb = tuple(nb)
                    if nb in closed or not free(nb):
                        continue
                    tentative = cur_g + float(np.linalg.norm(ee(nb) - cur_ee))
                    if tentative < g.get(nb, np.inf):
                        g[nb] = tentative
                        came[nb] = cur
                        heapq.heappush(open_heap, (tentative + h(nb), next(counter), nb))
        return fail("no path (open list exhausted)", expansions)
