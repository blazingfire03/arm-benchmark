import os
os.environ["NANOBIND_LEAK_WARNINGS"] = "0"

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from .base import Planner
from .utils import shortcut, densify
from ..core.types import Trajectory

PLANNERS = {
    "RRT": og.RRT,
    "RRT-Connect": og.RRTConnect,
    "RRT*": og.RRTstar,
    "PRM": og.PRM,
}


class OMPLPlanner(Planner):
    def __init__(self, algorithm="RRT-Connect", time_budget=5.0, shortcut_step=0.05,
                 simplify=True):
        self.algorithm = algorithm
        self.name = algorithm
        self.time_budget = time_budget
        self.shortcut_step = shortcut_step
        self.simplify = simplify

    def plan(self, world, start, goal) -> Trajectory:
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)

        space = ob.RealVectorStateSpace(7)
        bounds = ob.RealVectorBounds(7)
        for i in range(7):
            bounds.setLow(i, float(world.lower[i]))
            bounds.setHigh(i, float(world.upper[i]))
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)

        class Validity(ob.StateValidityChecker):
            def __init__(self, si_, w_):
                super().__init__(si_)
                self.w = w_
            def isValid(self, state):
                q = np.array([state[i] for i in range(7)])
                return bool(self.w.is_collision_free(q))

        si.setStateValidityChecker(Validity(si, world))
        si.setStateValidityCheckingResolution(0.01)
        si.setup()

        ss = si.allocState()
        gs = si.allocState()
        for i in range(7):
            ss[i] = float(start[i])
            gs[i] = float(goal[i])

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(ss, gs)

        planner_cls = PLANNERS.get(self.algorithm)
        if planner_cls is None:
            raise ValueError(f"unknown OMPL planner: {self.algorithm}")
        planner = planner_cls(si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        solved = planner.solve(self.time_budget)
        exact = pdef.hasExactSolution()

        if not exact:
            return Trajectory(waypoints=np.array([start, goal]), dt=world.dt,
                              planning_success=False,
                              metadata={"reason": "no exact solution",
                                        "approximate": bool(pdef.hasApproximateSolution()),
                                        "gap": float(pdef.getSolutionDifference()) if pdef.hasApproximateSolution() else None})

        path = pdef.getSolutionPath()
        if self.simplify:
            simplifier = og.PathSimplifier(si)
            simplifier.simplifyMax(path)

        n_states = path.getStateCount()
        corners = [start]
        for k in range(n_states):
            s = path.getState(k)
            corners.append(np.array([s[i] for i in range(7)]))
        corners.append(goal)

        sc = shortcut(world, corners, self.shortcut_step)
        dense = densify(sc, self.shortcut_step)

        return Trajectory(waypoints=dense, dt=world.dt, planning_success=True,
                          metadata={"ompl_states": n_states,
                                    "shortcut_corners": len(sc),
                                    "dense_waypoints": len(dense)})
