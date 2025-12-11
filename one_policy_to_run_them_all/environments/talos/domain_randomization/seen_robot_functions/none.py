class NoneDomainSeenRobotFunction:
    def __init__(self, env):
        self.env = env

    def init(self):
        self.seen_body_mass = self.env.model.body_mass.copy()
        self.seen_torque_limit = self.env.model.actuator_ctrlrange[:, 1].copy()
        self.seen_joint_nominal_position = self.env.nominal_joint_positions.copy()
        self.seen_joint_velocity = self.env.max_joint_velocities.copy()
        self.seen_joint_range = self.env.model.jnt_range[1:].copy()
        self.seen_joint_damping = self.env.model.dof_damping[6:].copy()
        self.seen_joint_armature = self.env.model.dof_armature[6:].copy()
        self.seen_joint_stiffness = self.env.model.jnt_stiffness[1:].copy()
        self.seen_joint_frictionloss = self.env.model.dof_frictionloss[6:].copy()

    def sample(self):
        return
