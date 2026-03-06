
import numpy as np
import aligator
import pinocchio as pin

from aligator_mpc.mpcParameters import(
    StateRegularizationWeights,
    CostsConfig,
    WaypointWeightsConfig,
    StateRegularizationWeights
)

class Costs():
    def __init__(self, parameters, robot, space, interpolator):
        self.parameters = parameters
        self.robot = robot
        self.space = space
        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv
        self.nu = self.nv
        self.ndx = self.space.ndx
        self.tool_id = self.robot.model.getFrameId(self.parameters.robot.tool_frame_name)
        self.interpolator = interpolator

        self.stage_dep_costs = []
        self.stage_indep_costs = []
        self.terminal_costs = []

        self.addCosts(self.parameters.task.mpc.costs)


    def addStateRegularizationCost(self,reg_cost:StateRegularizationWeights, terminal:bool) -> None:
        """Adds regulation costs to self.stage_dep_costs
        """
        # Handle the different modes for the state regularisation weights (scalar, scalable-vector)
        match reg_cost.position.mode:
            case "scalable-vector":
                weight_pos = np.array(reg_cost.position.values)*reg_cost.position.scale
            case "scalar":
                weight_pos = reg_cost.position.value * np.ones(self.nq)
        match reg_cost.velocity.mode:
            case "scalable-vector":
                weight_vel = np.array(reg_cost.velocity.values)*reg_cost.velocity.scale
            case "scalar":
                weight_vel = reg_cost.velocity.value * np.ones(self.nv)
        match reg_cost.torque.mode:
            case "scalable-vector":
                weight_torque = np.array(reg_cost.torque.values)*reg_cost.torque.scale
            case "scalar":
                weight_torque = reg_cost.torque.value * np.ones(self.nu)

        wt_x = np.diag(np.concatenate([weight_pos, weight_vel]))
        wt_u = np.diag(weight_torque)

        position_ref = self.parameters.task.mpc.regularisation_ref.joint_pos
        vel_ref = self.parameters.task.mpc.regularisation_ref.joint_vel
        x_ref = np.array(position_ref + vel_ref)

        if not terminal:
            # State reg
            stage_reg_cost = [(f"reg_state_{i}", aligator.QuadraticStateCost(self.space, self.nu, x_ref, wt_x)) for i in range(self.parameters.task.mpc.n_total_steps)]
            # Check if at least one of the weights is non-zero before adding the cost to the stage definition
            if(np.max(np.abs(weight_vel)) > 0 or np.max(np.abs(weight_pos)) > 0):
                self.stage_dep_costs.append(stage_reg_cost)
            # Control reg
            control_reg_cost = [(f"reg_ctrl_{i}", aligator.QuadraticControlCost(self.space, np.zeros(self.nu), wt_u)) for i in range(self.parameters.task.mpc.n_total_steps)]
            if(np.max(np.abs(weight_torque)) > 0):
                self.stage_dep_costs.append(control_reg_cost)
        else:
            # State reg
            if(np.max(np.abs(weight_vel)) > 0 or np.max(np.abs(weight_pos)) > 0):
                self.terminal_costs.append(("reg_state_term", aligator.QuadraticStateCost(self.space, self.nu, x_ref, wt_x)))
            # Control reg
            if(np.max(np.abs(weight_torque)) > 0):
                self.terminal_costs.append(("reg_ctrl_term", aligator.QuadraticControlCost(self.space, np.zeros(self.nu), wt_u)))

    def addWaypointCosts(self, cost:WaypointWeightsConfig) -> None:
        """For each stage adds a cost tied to matching the end effector frame to a waypoint frame and a cost tied to matching the frame velocity
        """
        pose_translation_weights, pose_orientation_weights = cost.pose.get_weights()
        vel_translation_weights, vel_orientation_weights = cost.velocity.get_weights()
        wt_frame_pose = np.diag(np.concatenate([pose_translation_weights, pose_orientation_weights]))
        wt_frame_vel = np.diag(np.concatenate([vel_translation_weights, vel_orientation_weights]))
        frame_vel_cost = []
        placement_costs = []
        for t in range (self.parameters.task.mpc.n_total_steps):
            pose , target_vel = self.interpolator(t*self.parameters.task.mpc.dt)
            # cost on the placement of the waypoint
            placement_residual = aligator.FramePlacementResidual(self.ndx, self.nu, self.robot.model, pose, self.robot.model.getFrameId(self.parameters.robot.tool_frame_name))
            cost = (f"pose_{t}", aligator.QuadraticResidualCost(self.space, placement_residual, wt_frame_pose))
            placement_costs.append(cost)

            # cost on the velocity of the waypoint
            frame_vel_fn = aligator.FrameVelocityResidual(self.ndx, self.nu, self.robot.model, target_vel, self.tool_id, pin.WORLD)
            cost_vel = (f"frame_vel_{t}", aligator.QuadraticResidualCost(self.space, frame_vel_fn, wt_frame_vel))
            frame_vel_cost.append(cost_vel)
        # Check if pose weights are non-zero
        pose_trans_max = np.max(np.abs(pose_translation_weights))
        pose_orient_max = np.max(np.abs(pose_orientation_weights))

        if (pose_trans_max > 0 or pose_orient_max > 0):
            self.stage_dep_costs.append(placement_costs)

        # Check if velocity weights are non-zero
        vel_trans_max = np.max(np.abs(vel_translation_weights))
        vel_orient_max = np.max(np.abs(vel_orientation_weights))

        if (vel_trans_max > 0 or vel_orient_max > 0):
            self.stage_dep_costs.append(frame_vel_cost)

    def addCosts(self, costs_list: CostsConfig) -> None:
        """Adds costs to self.stage_dep_costs using the `add` functions defined in this class according to the type of each cost in `costs_list`

        Args:
            costs_list (CostsConfig): list of costs to add
        """
        # Add running costs
        for _, cost in costs_list.running.items():
            if cost.enabled:
                match cost.weights.type:
                    case "state-regularisation":
                        self.addStateRegularizationCost(cost.weights, False)
                    case "waypoints":
                        self.addWaypointCosts(cost.weights)
                    case _:
                        raise ValueError(f"Cost type {cost.type} not recognized, available types are : 'regularisation' and 'waypoints'")
            else:
                continue
        # Add terminal costs
        for _, cost in costs_list.terminal.items():
            if cost.enabled:
                match cost.weights.type:
                    case "state-regularisation":
                        self.addStateRegularizationCost(cost.weights, True)
                    case _:
                        raise ValueError(f"Cost type {cost.type} not recognized, available types are : 'regularisation'")
            else:
                continue
