import numpy as np
import pybullet as p
import pybullet_data
from .types import Obstacle

HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


class PandaWorld:
    def __init__(self, gui=False, timestep=1/240, gravity=-9.81, self_collision=True,
                 table=True, acm_samples=1000, acm_thresh=0.95, acm_seed=0):
        self.dt = timestep
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, gravity, physicsClientId=self.cid)
        p.setTimeStep(timestep, physicsClientId=self.cid)

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.cid)

        self.table = None
        table_top = 0.0
        if table:
            self.table = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0],
                                    useFixedBase=True, physicsClientId=self.cid)
            tops = [p.getAABB(self.table, i, physicsClientId=self.cid)[1][2]
                    for i in range(-1, p.getNumJoints(self.table, physicsClientId=self.cid))]
            table_top = max(tops)
        self.base_pos = np.array([0.0, 0.0, table_top])

        flags = p.URDF_USE_SELF_COLLISION if self_collision else 0
        self.panda = p.loadURDF("franka_panda/panda.urdf", basePosition=list(self.base_pos),
                                useFixedBase=True, flags=flags, physicsClientId=self.cid)

        self._discover_joints()
        self._build_allowed(acm_samples, acm_thresh, acm_seed)
        self.obstacles = []
        self.reset_home()

    def _discover_joints(self):
        self.arm_joints, lower, upper = [], [], []
        self.flange_link = None
        for j in range(p.getNumJoints(self.panda, physicsClientId=self.cid)):
            info = p.getJointInfo(self.panda, j, physicsClientId=self.cid)
            jtype, name, child_link = info[2], info[1].decode(), info[12].decode()
            if child_link == "panda_link8":
                self.flange_link = j
            if jtype == p.JOINT_REVOLUTE and "finger" not in name and len(self.arm_joints) < 7:
                self.arm_joints.append(j)
                lower.append(info[8]); upper.append(info[9])
        self.lower, self.upper = np.array(lower), np.array(upper)
        if self.flange_link is None:
            self.flange_link = self.arm_joints[-1]

    def _build_allowed(self, n_samples, thresh, seed):
        self_allowed = set()
        for j in range(p.getNumJoints(self.panda, physicsClientId=self.cid)):
            parent = p.getJointInfo(self.panda, j, physicsClientId=self.cid)[16]
            self_allowed.add(frozenset((j, parent)))
        rng = np.random.default_rng(seed)
        self_counts, table_counts = {}, {}
        for _ in range(n_samples):
            self.set_config(rng.uniform(self.lower, self.upper))
            p.performCollisionDetection(physicsClientId=self.cid)
            for fp in {frozenset((c[3], c[4])) for c in
                       p.getContactPoints(bodyA=self.panda, bodyB=self.panda,
                                          physicsClientId=self.cid)}:
                self_counts[fp] = self_counts.get(fp, 0) + 1
            if self.table is not None:
                for l in {c[3] for c in p.getContactPoints(bodyA=self.panda, bodyB=self.table,
                                                           physicsClientId=self.cid)}:
                    table_counts[l] = table_counts.get(l, 0) + 1
        self.allowed_self_pairs = self_allowed | {fp for fp, ct in self_counts.items()
                                                  if ct / n_samples >= thresh}
        self.allowed_table_links = {l for l, ct in table_counts.items()
                                    if ct / n_samples >= thresh}

    def set_config(self, q):
        for idx, j in enumerate(self.arm_joints):
            p.resetJointState(self.panda, j, float(q[idx]), physicsClientId=self.cid)

    def reset_home(self):
        self.set_config(HOME)

    def contact_count(self, q):
        self.set_config(q)
        p.performCollisionDetection(physicsClientId=self.cid)
        n = 0
        for ob in self.obstacles:
            n += len(p.getContactPoints(bodyA=self.panda, bodyB=ob.pybullet_body_id,
                                        physicsClientId=self.cid))
        if self.table is not None:
            for c in p.getContactPoints(bodyA=self.panda, bodyB=self.table,
                                        physicsClientId=self.cid):
                if c[3] not in self.allowed_table_links:
                    n += 1
        for c in p.getContactPoints(bodyA=self.panda, bodyB=self.panda,
                                    physicsClientId=self.cid):
            if frozenset((c[3], c[4])) not in self.allowed_self_pairs:
                n += 1
        return n

    def is_collision_free(self, q):
        return self.contact_count(q) == 0

    def is_self_collision_free(self, q):
        self.set_config(q)
        p.performCollisionDetection(physicsClientId=self.cid)
        for c in p.getContactPoints(bodyA=self.panda, bodyB=self.panda, physicsClientId=self.cid):
            if frozenset((c[3], c[4])) not in self.allowed_self_pairs:
                return False
        return True

    def ee_position(self, q):
        self.set_config(q)
        return np.array(p.getLinkState(self.panda, self.flange_link,
                                       physicsClientId=self.cid)[4])

    def spawn(self, ob: Obstacle) -> int:
        if ob.shape_type == "box":
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(ob.dimensions),
                                         physicsClientId=self.cid)
        elif ob.shape_type == "cylinder":
            r, h = ob.dimensions
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(r), height=float(h),
                                         physicsClientId=self.cid)
        elif ob.shape_type == "sphere":
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=float(ob.dimensions[0]),
                                         physicsClientId=self.cid)
        else:
            raise ValueError(f"unknown shape: {ob.shape_type}")
        # colored visual so obstacles read clearly on the table
        colors = {"box": [0.9, 0.5, 0.1, 1], "cylinder": [0.2, 0.4, 1.0, 1],
                  "sphere": [0.85, 0.1, 0.1, 1]}
        if ob.shape_type == "box":
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(ob.dimensions),
                                      rgbaColor=colors["box"], physicsClientId=self.cid)
        elif ob.shape_type == "cylinder":
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=float(ob.dimensions[0]),
                                      length=float(ob.dimensions[1]), rgbaColor=colors["cylinder"],
                                      physicsClientId=self.cid)
        else:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(ob.dimensions[0]),
                                      rgbaColor=colors["sphere"], physicsClientId=self.cid)
        bid = p.createMultiBody(0, col, vis, basePosition=list(ob.position),
                                baseOrientation=list(ob.orientation), physicsClientId=self.cid)
        ob.pybullet_body_id = bid
        self.obstacles.append(ob)
        return bid

    def clear_obstacles(self):
        for ob in self.obstacles:
            p.removeBody(ob.pybullet_body_id, physicsClientId=self.cid)
        self.obstacles = []

    def disconnect(self):
        p.disconnect(self.cid)
