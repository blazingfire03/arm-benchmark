import os, numpy as np
os.environ["NANOBIND_LEAK_WARNINGS"] = "0"   # suppress nanobind noise

from ompl import base as ob
from ompl import geometric as og
from arm_benchmark.core.world import PandaWorld
from arm_benchmark.scenes.obstacles import load_obstacles
from arm_benchmark.scenes.testcases import load_test_cases

w = PandaWorld(gui=False)
load_obstacles(w, "config/sparse_obstacles.json")
cases = load_test_cases("config/sparse_test_cases.json")

# try multiple cases to find one that works cleanly
for ci in [6, 0, 1, 2, 3]:
    case = cases[ci]
    start, goal = np.array(case["start"]), np.array(case["goal"])
    print(f"\n--- case {ci} ---")
    print(f"  start valid: {w.is_collision_free(start)}")
    print(f"  goal valid:  {w.is_collision_free(goal)}")

    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    for i in range(7):
        bounds.setLow(i, float(w.lower[i]))
        bounds.setHigh(i, float(w.upper[i]))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)

    class Validity(ob.StateValidityChecker):
        def __init__(self, si, world):
            super().__init__(si)
            self.world = world
        def isValid(self, state):
            q = np.array([state[i] for i in range(7)])
            return bool(self.world.is_collision_free(q))

    si.setStateValidityChecker(Validity(si, w))
    si.setStateValidityCheckingResolution(0.02)
    si.setup()

    ss = si.allocState()
    gs = si.allocState()
    for i in range(7):
        ss[i] = float(start[i])
        gs[i] = float(goal[i])

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(ss, gs)

    for name, cls in [("RRTConnect", og.RRTConnect), ("RRT", og.RRT),
                      ("RRTstar", og.RRTstar), ("PRM", og.PRM)]:
        planner = cls(si)
        planner.setProblemDefinition(pdef)
        planner.setup()
        solved = planner.solve(5.0)
        exact = pdef.hasExactSolution()
        approx = pdef.hasApproximateSolution()
        if exact:
            path = pdef.getSolutionPath()
            sN = path.getState(path.getStateCount() - 1)
            goal_err = max(abs(sN[i] - goal[i]) for i in range(7))
            print(f"  {name:12s}: EXACT  | states={path.getStateCount():>4} | goal_err={goal_err:.6f}")
        elif approx:
            print(f"  {name:12s}: APPROX | gap={pdef.getSolutionDifference():.4f}")
        else:
            print(f"  {name:12s}: FAILED")
        pdef.clearSolutionPaths()
        planner.clear()

    if ci == 6:
        continue  # try others too
    break

w.disconnect()
