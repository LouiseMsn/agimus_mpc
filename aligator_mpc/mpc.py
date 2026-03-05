from aligator_mpc.mpcParameters import Config, StateRegularizationWeights, TorqueLimitsConstraint, CollisionConstraint, ConstraintType, CostsConfig, WaypointWeightsConfig, StateRegularizationWeights 
from aligator_mpc.mpcTrajectoryUtils import Interpolator
from aligator_mpc.mpcUtils import getIndexesFromJointNames
import aligator
from aligator import constraints, manifolds, dynamics
import pinocchio as pin
import numpy as np
from typing import List
import time
from aligator_mpc.mpcUtils import StagesDefinition

from ament_index_python.packages import get_package_share_directory


from rclpy.impl import rcutils_logger

class MPC():
    def __init__(self, waypoints : list, parameters : Config, robot_urdf : str) -> None:
        self.parameters = parameters
        self.waypoints = waypoints
        print(self.parameters)
        # Build the robot model
        model = pin.buildModelFromXML(robot_urdf)
        visual_model = pin.buildGeomFromUrdfString(model, robot_urdf, pin.GeometryType.VISUAL,  self.parameters.robot.meshes_packages)
        collision_model = pin.buildGeomFromUrdfString(model, robot_urdf, pin.GeometryType.COLLISION,  self.parameters.robot.meshes_packages)
        self.robot = pin.RobotWrapper(model, collision_model, visual_model)
        # Lock joints if needed
        self.robot.model = pin.buildReducedModel(self.robot.model, getIndexesFromJointNames(self.robot, self.parameters.robot.locked_joints), pin.neutral(self.robot.model))
        self.robot.data = self.robot.model.createData()        
        
        self.space = self.space = manifolds.MultibodyPhaseSpace(self.robot.model)
        self.tool_id = self.robot.model.getFrameId(self.parameters.robot.tool_frame_name)
        self.world_frame_id = self.robot.model.getFrameId(self.parameters.robot.world_frame_name)
        self.world_joint_id = self.robot.model.frames[self.world_frame_id].parentJoint

        self.ndx = self.space.ndx # size of state vector
        self.nq = self.robot.model.nq # size of joint state vector
        self.nv = self.robot.model.nv # size of speed vector
        self.nu = self.nv
        self.x0 = self.space.neutral() # initial robot state
        self.q0 = self.x0[:self.nq] # initial joints state

        
        pin.forwardKinematics(self.robot.model, self.robot.data, self.q0)
        pin.updateFramePlacements(self.robot.model, self.robot.data) # update model placemement

        self.discrete_dynamics = self.calcDiscreteDynamics()
        self.stage_factory = None # to be instanciated with start Pose

        # Solver
        self.solver, self.callback = self.instantiateSolver()
        self.results = None
        self.problem = None
        self.solver_stage_number = 0

    def instantiateSolver(self) -> tuple[aligator.SolverProxDDP, aligator.HistoryCallback]:
        """Creates the `SolverProxDDP solver` object and assigns its rollout type, sa strategy, linear solver choice and if using parallel aligator the number of threads.

        Returns:
            solver (aligator.SolverProxDDP)
            callback (aligator.HistoryCallback) 
        """

        solver = aligator.SolverProxDDP(self.parameters.task.mpc.solver.presolve.tolerance, self.parameters.task.mpc.solver.presolve.mu_init, max_iters=self.parameters.task.mpc.solver.presolve.max_iters, verbose=eval(self.parameters.task.mpc.solver.verbose))
        solver.rollout_type = eval(self.parameters.task.mpc.solver.rollout_type)
        solver.sa_strategy = eval(self.parameters.task.mpc.solver.sa_strategy)
        solver.linear_solver_choice = eval(self.parameters.task.mpc.solver.linear_solver_choice)
        if solver.linear_solver_choice == aligator.LQ_SOLVER_PARALLEL:
            solver.setNumThreads(self.parameters.task.mpc.solver.num_threads)
        callback = aligator.HistoryCallback(solver)
        solver.registerCallback("his", callback)

        return solver, callback
    
    def setStartPose(self, start_pose) -> None:
        """ Updates the pinocchio frames to the starting pose

        Args:
            start_pose (List[float]): starting pose of the robot

        Raises:
            AssertionError: If the starting pose is not of length nq
        """

        if len(start_pose)!=self.nq:
            raise AssertionError(f"Pose has the wrong number of elements: is {len(start_pose)} but should be {self.nq}")
        self.x0[:self.nq] = start_pose
        pin.forwardKinematics(self.robot.model, self.robot.data, start_pose)
        pin.updateFramePlacements(self.robot.model, self.robot.data) # update model placemement

    def initStages(self) -> None:
        """ Initializes the stage_factory object and the u_min and u_max variables
        """

        self.stage_factory = StageFactory(self.robot, self.space, self.parameters.task.mpc.n_total_steps, self.discrete_dynamics, self.waypoints, self.parameters)

        # Min & Max torque on command output
        self.u_min = self.stage_factory.u_min
        self.u_max = self.stage_factory.u_max

    def iterate(self, current_xs) -> float:
        """
        Runs the solver over one iteration
        Args:
            current_xs: (list) current state
        Returns:
            solver_calc_time (secs)
        """

        if self.stage_factory is None:
            raise ValueError('The stage factory has not been instanciated, use setStartPose to set a start pose and instanciate the stage factory')


        if self.solver_stage_number == 0:
            # create the data
            us = [self.computeQuasistatic(self.robot.model, self.x0, a = np.zeros(self.nv)) for _ in range(self.parameters.task.mpc.nb_steps_horizon)]
            xs = aligator.rollout(self.discrete_dynamics, self.x0, us)

            # create the stages & problem
            stages, terminal_coststack = self.stage_factory.fabricateStages(0, self.parameters.task.mpc.nb_steps_horizon)
            # terminal_coststack = self.stage_factory.getTerminalCoststack(0)

            self.problem = aligator.TrajOptProblem(self.x0, stages, terminal_coststack)
            self.solver.setup(self.problem)
            self.solver.max_iters = self.parameters.task.mpc.solver.presolve.max_iters
            self.solver.mu_init = self.parameters.task.mpc.solver.presolve.mu_init
            self.solver.tol = self.parameters.task.mpc.solver.presolve.tolerance            
        else:

            self.solver.max_iters = self.parameters.task.mpc.solver.running.max_iters
            self.solver.mu_init = self.parameters.task.mpc.solver.running.mu_init
            self.solver.tol = self.parameters.task.mpc.solver.running.tolerance
            # cycle the data
            us   = self.cycleData(self.results.us.tolist(), None, None)
            xs   = self.cycleData(self.results.xs.tolist(), current_xs,"xs")
            end_of_horizon_index = self.solver_stage_number + self.parameters.task.mpc.nb_steps_horizon # -1 because the first stage is 0

            # cycle the stages
            stage_model = self.stage_factory.getStageModel(end_of_horizon_index)
            self.problem.replaceStageCircular(stage_model)

            stage_data = stage_model.createData()   
            
            self.problem.x0_init = current_xs
            self.solver.cycleProblem(self.problem, stage_data)

        self.results, solver_calc_time = self.run_solver(self.problem, us=us, xs=xs)
        self.solver_stage_number += 1

        return solver_calc_time

    def run_solver(self, problem, *, us, xs) -> tuple[aligator.Results, float]:
        """
        Runs the solver with xs and us warmstart

        Args:
            problem (aligator.TrajOptProblem): aligator problem
            us (List[float]): warmstart on command
            xs (List[float]): warmstart on state

        Returns:
            results (aligator.Results): results of the solvers' iteration
            timer (float) : solver calculation time
        """

        start = time.time()
        self.solver.run(problem, xs, us)
        end = time.time()
        results = self.solver.results
        timer = end - start

        return results, timer

    def calcDiscreteDynamics(self) -> aligator.dynamics.IntegratorSemiImplEuler:
        """Initializes the discrete dynamic of the system

        Returns:
            dynamic_model (aligator.dynamics.IntegratorSemiImplEuler)
        """

        nu = self.robot.model.nv
        B_mat = np.eye(nu)
        ode = dynamics.MultibodyFreeFwdDynamics(self.space, B_mat) # Ordinatry Diff Equation: resolution de l'équation de la dynamique
        dynamic_model = dynamics.IntegratorSemiImplEuler(ode, self.parameters.task.mpc.dt)
        return dynamic_model

    def computeQuasistatic(self, robot_model: pin.Model, x0, a) -> list[float]:
        """Initializes individual us values 

        Args:
            robot_model (pin.Model): robot model
            x0 (List[float]): starting state
            a (np.arrray): joint acceleration vector

        Returns:
           joint torque List[float]
        """

        data = robot_model.createData()
        q0 = x0[:self.nq]
        v0 = x0[self.nq : self.nq + self.nv]
        joint_torque = pin.rnea(robot_model, data, q0, v0, a) # inverse dynamics given q, v and a
        return joint_torque

    def cycleData(self, list:List, current_x:List, type:str)-> List:
        """Used during mpc iteration, thrashes the first item of the list and adds a copy of the last item to its end.
        \If type is xs then return[0] is overwritten to current_x

        Args:
            list (List): list to cycle
            current_x (List): state to overwritte to return[0]
            type (str): type of the list

        Returns:
            list(List): cycled list
        """

        list = list[1:]
        list.append(list[-1])
        if type =="xs":
            list[0] = current_x
        
        return list

    def get_endpoint_traj(self, states: List[np.ndarray]) -> np.array:
        """Gets the trajectory of the end effector for a list of states

        Args:
            states (List[np.ndarray]): list of states

        Returns:
            np.array: list of end effector positions
        """

        pts = []
        for i in range(len(states)):
            pts.append(self.get_endpoint(states[i][: self.nq]))
        return np.array(pts)

    def get_endpoint(self, q: np.ndarray) -> list[float]:
        """Gets the effector pose for a joint configuration q

        Args:
            q (np.ndarray): joint configuration

        Returns:
            list : position of the end effector
        """

        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
        ee_pos = self.robot.data.oMf[self.tool_id].translation.copy()
        return ee_pos

class StageFactory():
    def __init__(self, robot, space, n_steps, discrete_dynamics, waypoints, params: Config) -> None:
        self.robot = robot
        self.space = space
        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv
        self.nu = self.nv
        self.ndx = self.space.ndx
        self.u_max = self.robot.model.effortLimit
        self.u_min = -self.u_max
        self.discrete_dynamics = discrete_dynamics
        self.n_steps = n_steps
        self.parameters = params
        self.tool_id = self.robot.model.getFrameId(self.parameters.robot.tool_frame_name)

        self.stages_definition = StagesDefinition()
        self.stages : list[aligator.stageModel] = []
        self.stage_model_list = []
        self.problem : aligator.TrajOptProblem

        self.waypoints = waypoints

        self.interpolator = self.getInterpolator()

        self.addConstraints(self.parameters.task.mpc.constraints)
        self.addCosts(self.parameters.task.mpc.costs)
        self.buildStageModelList()

    def getInterpolator(self) -> Interpolator:
        """Builds and returns the interpolator object

        Returns:
            Interpolator: waypoint interpolator
        """
        start_pos = self.robot.data.oMf[self.tool_id]        
        self.waypoints = [start_pos] + self.waypoints
        rcutils_logger.RcutilsLogger(name="   MPC_DEBUG   ").info(f'start {pin.rpy.matrixToRpy(start_pos.rotation)} { start_pos.translation}')
        return Interpolator(self.waypoints, self.parameters.task.trajectory.vel)

    def getStageModel(self, stage_number:int) -> aligator.StageModel :
        """Returns the stage model for the `stage_number` th stage

        Args:
            stage_number (int): index of the stage to get
        Returns:
            aligator.StageModel : stage model
        """

        if len(self.stage_model_list ) == 0:
            raise ValueError("The stage model list is not built, run buildStageModelList() before running getStageModel()")
        else:
            if stage_number >= self.parameters.task.mpc.n_total_steps :
                return self.stage_model_list[-1]
            else:
                return self.stage_model_list[stage_number]
        
    def buildStageModelList(self) -> aligator.StageModel:
        """Builds and returns the StageModel for the `stage_number` th stage of the problem.

        Returns:
            List[aligator.StageModel]: list of stage models 
        """

        for stage_number in range(self.parameters.task.mpc.n_total_steps):
        
            stage_coststack = aligator.CostStack(self.space, self.nu)
            cost_list = self.getDynamicCosts(stage_number, self.stages_definition.stage_dep_costs)
            for cost in cost_list:
                stage_coststack.addCost(*cost)

            stage_model = aligator.StageModel(stage_coststack, self.discrete_dynamics)
            for constraint in self.stages_definition.constraints:
                stage_model.addConstraint(*constraint)
            self.stage_model_list.append(stage_model)
        
        return stage_model

    def getTerminalCoststack(self, stage_number:int) -> aligator.CostStack:
        """Returns the terminal coststack calculated from the `stages_definitions` dict.

        Args:
            stage_number (int): index of the stage

        Returns:
            aligator.CostStack: terminal coststack
        """

        terminal_coststack = aligator.CostStack(self.space, self.nu)
        cost_list = self.getDynamicCosts(stage_number, self.stages_definition.terminal_costs)
        for cost in cost_list:
            terminal_coststack.addCost(*cost)

        return terminal_coststack

    def fabricateStages(self, current_stage:int, duration:int) -> tuple[list, aligator.CostStack]:
        """Builds the stages from `current_stage` to `duration` using the costs stored in `self.stages_definition`

        Args:
            current_stage (int): index of the current stage
            duration (_int): duration (in number of indexes)

        Returns:
            List[aligator.StageModel] : list of stages model from index to (index + duration - 1)
            aligator.CostStack : terminal coststack
        """

        terminal_coststack = aligator.CostStack(self.space, self.nu)
        for terminal_cost in self.stages_definition.terminal_costs:
            terminal_coststack.addCost(*terminal_cost)

        stages = []
        for stage_num in range(current_stage, duration + current_stage):
            stage_coststack = aligator.CostStack(self.space, self.nu)
            cost_list = self.getDynamicCosts(stage_num, self.stages_definition.stage_dep_costs)
            if len(self.stages_definition.stage_indep_costs) != 0:
                cost_list = cost_list + self.stages_definition.stage_indep_costs
            for cost in cost_list:
                stage_coststack.addCost(*cost)

            stage_model = aligator.StageModel(stage_coststack, self.discrete_dynamics)
            for constraint in self.stages_definition.constraints:
                stage_model.addConstraint(*constraint)
            stages.append(stage_model)
        return stages , terminal_coststack

    # ==========================================================================
    # Constraints functions
    # ==========================================================================
    def addJointsLimitsConstraints(self) -> None:
        """Adds joints limits constraints (joint_angle_residual, box_constraint) to self.stages_definition.constraints
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
            self.stages_definition.constraints.append(constraint)
    
    def addVelocityLimitsConstraints(self) -> None :
        # TODO : implement velocity limits constraints
        pass

    def addTorqueLimitsConstraints(self, constraint: TorqueLimitsConstraint) -> None:
        """Adds torque limits constraints (joint_angle_residual, box_constraint) to self.stages_definition.constraints 
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
        nv = self.robot.model.nv
        nu = nv
        ndx = self.space.ndx
        residual = aligator.ControlErrorResidual(ndx, nu)
        constraint = constraints.BoxConstraint(self.u_min, self.u_max)
        self.stages_definition.constraints.append((residual, constraint))

    def addCollisionsConstraints(self, collision_constraint: CollisionConstraint) -> None: # TODO : implement self collision constraints using the collision pairs defined in `collision_constraint` and the distance function of pinocchio, add the possibility to use a custom distance function
        """
        Adds a cost in `self.stages_definition.constraints` linked to self collisions of the robot.
        \n Work in progress
        """

        pairs_2_add = [23,35]
        for i in pairs_2_add:
            collision_residual = aligator.FrameCollisionResidual(self.ndx, self.nu, self.robot.model, self.robot.collision_model, i)
            # log barrier : weight*ln(function)
            self.stages_definition.stage_indep_costs.append((f"collision_{i}", aligator.LogResidualCost(self.space, collision_residual, self.parameters.task.mpc.weights.collision * np.eye(collision_residual.nr))))

            collision_constraint = constraints.BoxConstraint(np.array([0.05]), np.array([100]))
            self.stages_definition.constraints.append((collision_residual, collision_constraint))

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
    # ==========================================================================
    # Costs functions
    # ==========================================================================
    def addStateRegularizationCost(self,reg_cost:StateRegularizationWeights, terminal:bool) -> None:
        """Adds regulation costs to self.stages_definition.stage_dep_costs
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
                self.stages_definition.stage_dep_costs.append(stage_reg_cost)
            # Control reg
            control_reg_cost = [(f"reg_ctrl_{i}", aligator.QuadraticControlCost(self.space, np.zeros(self.nu), wt_u)) for i in range(self.parameters.task.mpc.n_total_steps)]
            if(np.max(np.abs(weight_torque)) > 0):
                self.stages_definition.stage_dep_costs.append(control_reg_cost)
        else:
            # State reg
            if(np.max(np.abs(weight_vel)) > 0 or np.max(np.abs(weight_pos)) > 0):
                self.stages_definition.terminal_costs.append(("reg_state_term", aligator.QuadraticStateCost(self.space, self.nu, x_ref, wt_x)))
            # Control reg
            if(np.max(np.abs(weight_torque)) > 0):
                self.stages_definition.terminal_costs.append(("reg_ctrl_term", aligator.QuadraticControlCost(self.space, np.zeros(self.nu), wt_u)))

    def addWaypointCosts(self, cost:WaypointWeightsConfig) -> None:
        """For each stage adds a cost tied to matching the end effector frame to a waypoint frame and a cost tied to matching the frame velocity
        """
        pose_translation_weights,pose_orientation_weights = cost.pose.get_weights()
        vel_translation_weights,vel_orientation_weights = cost.velocity.get_weights()
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
            self.stages_definition.stage_dep_costs.append(placement_costs)
        
        # Check if velocity weights are non-zero
        vel_trans_max = np.max(np.abs(vel_translation_weights))
        vel_orient_max = np.max(np.abs(vel_orientation_weights))
        
        if (vel_trans_max > 0 or vel_orient_max > 0):
            self.stages_definition.stage_dep_costs.append(frame_vel_cost)

    def addCosts(self, costs_list: CostsConfig) -> None:
        """Adds costs to self.stages_definition.stage_dep_costs using the `add` functions defined in this class according to the type of each cost in `costs_list`

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
    # ==========================================================================
    # Utils
    # ==========================================================================

    def getDynamicCosts(self, stage_number:int, master_cost_list:list) -> list:
        """Returns a list of costs for the current stage. If a cost is not defined for a given `stage_number`, the last instance of this cost is returned instead.

        Args:
            stage_number (int): index of the stage's cost
            master_cost_list (list): list where all the costs are stored

        Returns:
            list: list of costs to apply during stage_number
        """
        return_costs_list = []
        for cost_list in master_cost_list : 
            if stage_number < len(cost_list):
                return_costs_list.append(cost_list[stage_number])
            else:
                return_costs_list.append(cost_list[-1])
        return return_costs_list

    def getFullTrajectory_pt_by_pt(self) -> list:  # TODO : move to pattern generator?
        """Return the full trajectory formatted as np.array([x0,y0,z0], [x1,y1,z1], ...)

        Returns:
            list: list of points
        """
        traj = []
        for i in range(self.parameters.task.mpc.n_total_steps):
            pos, _ = self.interpolator(i*self.parameters.task.mpc.dt)
            traj.append(pos.translation)
        return traj
