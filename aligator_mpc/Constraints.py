import aligator
from aligator import constraints

from aligator_mpc.mpcParameters import(
    TorqueLimitsConstraint,
    CollisionConstraint,
    ConstraintType,
)
from aligator_mpc.mpcUtils import getIndexesFromJointNames

from typing import List
import numpy as np

class Constraints():
    def __init__(self, parameters, robot, space):
        self.parameters = parameters
        self.robot = robot
        self.space = space
        self.nv = self.robot.model.nv
        self.nu = self.nv
        self.ndx = self.space.ndx
        self.u_max = self.robot.model.effortLimit
        self.u_min = -self.u_max
        self.constraints = []

    def addJointsLimitsConstraints(self) -> None:
        """Adds joints limits constraints (joint_angle_residual, box_constraint) to self.constraints
        """

        # Base residual that extracts the state (f(x) = x - 0)
        identity_residual = aligator.StateErrorResidual(self.space, self.nu, self.space.neutral())

        for j_idx, jn in enumerate(self.robot.model.names[1:]):
            q_idx_in_x = self.robot.model.joints[j_idx + 1].idx_q
            q_min = self.robot.model.lowerPositionLimit[q_idx_in_x]
            q_max = self.robot.model.upperPositionLimit[q_idx_in_x]

            if np.isneginf(q_min) and np.isinf(q_max):
                continue

            A = np.zeros((1, self.ndx))
            A[0, q_idx_in_x] = 1.0
            b = np.array([0.0])
            joint_angle_residual = aligator.LinearFunctionComposition(identity_residual, A, b)

            box_constraint = constraints.BoxConstraint(np.array([q_min]), np.array([q_max]))

            constraint = (joint_angle_residual, box_constraint)
            self.constraints.append(constraint)

    def addVelocityLimitsConstraints(self) -> None :
        # TODO : implement velocity limits constraints
        pass

    def addTorqueLimitsConstraints(self, constraint: TorqueLimitsConstraint) -> None:
        """Adds torque limits constraints (joint_angle_residual, box_constraint) to self.constraints
        using the limits defined in `constraint` and applying the scaling defined in `constraint` to the default torque limits of the robot.

        Parameters:
            constraint (TorqueLimitsConstraint): contains the min and max torque limits to apply.
        """
        if constraint.per_joint_scaling:
            for joint, scaling in constraint.per_joint_scaling:
                joint_idx = getIndexesFromJointNames(self.robot, [joint])[0]
                self.u_max[joint_idx] *= scaling
                self.u_min[joint_idx] *= scaling
        else:
            self.u_max = np.array(self.u_max) * np.array(constraint.scale_factor)
            self.u_min = np.array(self.u_min) * np.array(constraint.scale_factor)

        residual = aligator.ControlErrorResidual(self.ndx, self.nu)
        constraint = constraints.BoxConstraint(self.u_min, self.u_max)
        self.constraints.append((residual, constraint))

    def addCollisionsConstraints(self, collision_constraint: CollisionConstraint) -> None: # TODO : implement self collision constraints using the collision pairs defined in `collision_constraint` and the distance function of pinocchio, add the possibility to use a custom distance function
        """
        Adds a cost in `self.constraints` linked to self collisions of the robot.
        \n Work in progress
        """

        #TODO TEST BETWEEN COST AND CONSTRAINT, WHY IS THERE CURRENTLY TWO IMPLEMs
        # pairs_2_add = [23,35]
        # for i in pairs_2_add:
        #     collision_residual = aligator.FrameCollisionResidual(self.ndx, self.nu, self.robot.model, self.robot.collision_model, i)
        #     # log barrier : weight*ln(function)
        #     self.stages_definition.stage_indep_costs.append((f"collision_{i}", aligator.LogResidualCost(self.space, collision_residual, self.parameters.task.mpc.weights.collision * np.eye(collision_residual.nr))))

        #     collision_constraint = constraints.BoxConstraint(np.array([0.05]), np.array([100]))
        #     self.stages_definition.constraints.append((collision_residual, collision_constraint))

    def addConstraints(self, constraints_list: List[ConstraintType]) -> None:
        """Adds constraints to self.stages_definition.constraints using the `add` functions defined in this class according to the type of each constraint in `constraints_list`

        Args:
            constraints_list (List[ConstraintType]): list of constraints to add
        """
        for constraint in constraints_list:
            if constraint.enabled:
                match constraint.type:
                    case "joint_limits":
                        self.addJointsLimitsConstraints()
                    case "velocity_limits":
                        # self.addVelocityLimitsConstraints()
                        pass
                    case "torque_limits":
                        self.addTorqueLimitsConstraints(constraint)
                    case "collision":
                        # self.addCollisionsConstraints(constraint)
                        pass
                    case _:
                        raise ValueError(f"Constraint type {constraint.type} not recognized, available types are : 'joint_limits', 'velocity_limits', 'torque_limits' and 'collision'")
