from aligator_mpc.mpcTrajectoryUtils import Interpolator

import aligator
from aligator import constraints, manifolds, dynamics
import example_robot_data as ex_robot_data
import pinocchio as pin
import numpy as np
from typing import List
import time
import math
from aligator_mpc.mpcUtils import StagesDefinition
from pinocchio.visualize import MeshcatVisualizer
from copy import copy
import sys
from ament_index_python.packages import get_package_share_directory

# !! TEMP
from line_profiler import profile
from rclpy.impl import rcutils_logger

class MPC():
    def __init__(self, waypoints, parameters):
        self.parameters = parameters
        self.waypoints = waypoints
        print(self.parameters)
        # Initialize robot
        # self.robot = ex_robot_data.load(self.parameters.robot.name)
        self.robot = r = pin.RobotWrapper.BuildFromURDF("src/agimus-demos/agimus_demo_09_glue_spreading/urdf/fr3.urdf",package_dirs=[get_package_share_directory("franka_description")])

        self.robot.model = pin.buildReducedModel(self.robot.model, [8,9], pin.neutral(self.robot.model))
        self.robot.data = self.robot.model.createData()

        self.space = self.space = manifolds.MultibodyPhaseSpace(self.robot.model)
        self.tool_id = self.robot.model.getFrameId(self.parameters.robot.tool_frame_name)
        self.world_frame_id = self.robot.model.getFrameId(self.parameters.robot.world_frame_name)
        self.world_joint_id = self.robot.model.frames[self.world_frame_id].parentJoint

        self.n_dx = self.space.ndx # size of state vector
        self.n_q = self.robot.model.nq # size of joint state vector
        self.n_v = self.robot.model.nv # size of speed vector
        self.nu = self.n_v
        self.x0 = self.space.neutral() # initial robot state
        self.q0 = self.x0[:self.n_q] # initial joints state

        
        pin.forwardKinematics(self.robot.model, self.robot.data, self.q0)
        pin.updateFramePlacements(self.robot.model, self.robot.data) # update model placemement

        self.discrete_dynamics = self.calcDiscreteDynamics()
        self.stage_factory = None # to be instanciated with start Pose

        # Solver
        self.solver, self.callback = self.instantiateSolver()
        self.results = None
        self.problem = None
        self.solver_stage_number = 0

    def instantiateSolver(self):
        """
        Creates the `SolverProxDDP solver` object and assigns its rollout type, sa strategy, linear solver choice and if using parallel aligator the number of threads.
        Sets the `self.callback` and `self.solver` object.
        """
        solver = aligator.SolverProxDDP(self.parameters.mpc.solver.presolve.tolerance, self.parameters.mpc.solver.presolve.mu_init, max_iters=self.parameters.mpc.solver.presolve.max_iters, verbose=eval(self.parameters.mpc.solver.verbose))
        solver.rollout_type = eval(self.parameters.mpc.solver.rollout_type)
        solver.sa_strategy = eval(self.parameters.mpc.solver.sa_strategy)
        solver.linear_solver_choice = eval(self.parameters.mpc.solver.linear_solver_choice)
        if solver.linear_solver_choice == aligator.LQ_SOLVER_PARALLEL:
            solver.setNumThreads(self.parameters.mpc.solver.num_threads)
        callback = aligator.HistoryCallback(solver)
        solver.registerCallback("his", callback)

        return solver, callback
    
    def setStartPose(self, start_pose):
        if len(start_pose)!=self.n_q:
            raise AssertionError(f"Pose has the wrong number of elements: is {len(start_pose)} but should be {self.n_q}")
        self.x0[:self.n_q] = start_pose
        pin.forwardKinematics(self.robot.model, self.robot.data, start_pose)
        pin.updateFramePlacements(self.robot.model, self.robot.data) # update model placemement

    def initStages(self):
        """
        Initializes the stage_factory object
        """
        self.stage_factory = StageFactory(self.robot, self.space, self.parameters.mpc.n_total_steps, self.discrete_dynamics, self.waypoints, self.parameters)

        # Min & Max torque on command output
        self.u_min = self.stage_factory.u_min
        self.u_max = self.stage_factory.u_max

    def iterate(self, current_xs):
        """
        Runs the solver over one iteration
        Args:
            current_xs: (list) current state
        Returns:
            solver_calc_time (secs)
        """

        if self.stage_factory is None:
            raise ValueError('The stage factory has not been instanciated, use setStartPose to set a start pose and instanciate the stage factory')

        # rcutils_logger.RcutilsLogger(name="   MPC_DEBUG   ").info(f'stage number { self.solver_stage_number}')

        if self.solver_stage_number == 0:
            # create the data
            us = [self.computeQuasistatic(self.robot.model, self.x0, a = np.zeros(self.n_v)) for _ in range(self.parameters.mpc.nb_steps_horizon)]
            xs = aligator.rollout(self.discrete_dynamics, self.x0, us)

            # create the stages & problem
            stages, terminal_coststack = self.stage_factory.fabricateStages(0, self.parameters.mpc.nb_steps_horizon)
            # terminal_coststack = self.stage_factory.getTerminalCoststack(0)

            self.problem = aligator.TrajOptProblem(self.x0, stages, terminal_coststack)
            self.solver.setup(self.problem)
            self.solver.max_iters = self.parameters.mpc.solver.presolve.max_iters
            self.solver.mu_init = self.parameters.mpc.solver.presolve.mu_init
            self.solver.tol = self.parameters.mpc.solver.presolve.tolerance            
        else:

            self.solver.max_iters = self.parameters.mpc.solver.running.max_iters
            self.solver.mu_init = self.parameters.mpc.solver.running.mu_init
            self.solver.tol = self.parameters.mpc.solver.running.tolerance
            # cycle the data
            us   = self.cycleData(self.results.us.tolist(), None, None)
            xs   = self.cycleData(self.results.xs.tolist(), current_xs,"xs")
            end_of_horizon_index = self.solver_stage_number + self.parameters.mpc.nb_steps_horizon # -1 because the first stage is 0

            # cycle the stages
            stage_model = self.stage_factory.getStageModel(end_of_horizon_index)
            self.problem.replaceStageCircular(stage_model)

            stage_data = stage_model.createData()   
            
            self.problem.x0_init = current_xs
            self.solver.cycleProblem(self.problem, stage_data)

            # self.solver.setup(self.problem)

        self.results, solver_calc_time = self.run_solver(self.problem, us=us, xs=xs)

        self.solver_stage_number += 1

        return solver_calc_time

    def run_solver(self, problem, *, us, xs, lams=None, vs=None):
        """
        Runs the solver
        """
        start = time.time()
        self.solver.run(problem, xs, us) # remove warmstart
        end = time.time()
        results = self.solver.results
        timer = end - start

        return results, timer

    def calcDiscreteDynamics(self):
        """
        Initializes the discrete dynamic of the system.
        """
        nu = self.robot.model.nv
        B_mat = np.eye(nu)
        ode = dynamics.MultibodyFreeFwdDynamics(self.space, B_mat) # Ordinatry Diff Equation: resolution de l'équation de la dynamique
        return dynamics.IntegratorSemiImplEuler(ode, self.parameters.mpc.dt)

    def computeQuasistatic(self, model: pin.Model, x0, a):
        """
        Initializes individual us values.
        """
        data = model.createData()
        q0 = x0[:self.n_q]
        v0 = x0[self.n_q : self.n_q + self.n_v]

        return pin.rnea(model, data, q0, v0, a)

    def cycleData(self, list:List, current_xs:List, type:str)-> List:
        """
        Used during mpc iteration, thrashes the first item of the list and adds a copy of the last item to its end
        """
        list = list[1:]
        list.append(list[-1])
        if type =="xs":
            list[0] = current_xs
        
        return list

    def get_endpoint_traj(self, xs: List[np.ndarray]):
        """
        Gets the trajectory of the effector for a state list
        """
        pts = []
        for i in range(len(xs)):
            pts.append(self.get_endpoint(xs[i][: self.n_q]))
        return np.array(pts)

    def get_endpoint(self, q: np.ndarray):
        """
        Gets the effector pose for a joint configuration q
        """
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
        return self.robot.data.oMf[self.tool_id].translation.copy()


class StageFactory():
    def __init__(self, robot, space, n_steps, discrete_dynamics, waypoints, params):
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

        # self.stages_definition: dict[str,list[tuple|list]] = {"constraints":[], "terminal costs":[], "stage dependant costs":[], "stage independant costs":[]} # Constraints are not stage-dependant
        self.stages_definition = StagesDefinition()
        self.stages : list[aligator.stageModel] = []
        self.stage_model_list = []
        self.problem : aligator.TrajOptProblem

        self.waypoints = waypoints

        self.interpolator = self.getInterpolator()

        self.addWaypointCosts()
        self.addOrientationCosts()
        self.addJointsLimitsConstraints()
        self.addTorqueLimitsConstraints()
        self.addRegulationCosts()
        # self.addAutoCollisionsConstraints()
        self.buildStageModelList()

    def getInterpolator(self):
        tool_id = self.robot.model.getFrameId(self.parameters.robot.tool_frame_name)
        start_pos = self.robot.data.oMf[tool_id]        
        self.waypoints = [start_pos] + self.waypoints
        rcutils_logger.RcutilsLogger(name="   MPC_DEBUG   ").info(f'start {pin.rpy.matrixToRpy(start_pos.rotation)} { start_pos.translation}')
        return Interpolator(self.waypoints, self.parameters.trajectory.vel)

    def getStageModel(self, stage_number):
        """Returns the stage model for the _stage_number_ th stage

        Args:
            stage_number (int): _description_
        """
        if len(self.stage_model_list ) == 0:
            raise ValueError("The stage model list is not built, run buildStageModelList() before running getStageModel()")
        else:
            if stage_number >= self.parameters.mpc.n_total_steps :
                # rcutils_logger.RcutilsLogger(name="   MPC_DEBUG   ").info('fin de trajectoire')
                return self.stage_model_list[-1]
            else:
                return self.stage_model_list[stage_number]
        
    def buildStageModelList(self):
        """
        Builds and returns the StageModel for the `stage_number` th stage of the problem.
        """
        for stage_number in range(self.parameters.mpc.n_total_steps):
        
            stage_coststack = aligator.CostStack(self.space, self.nu)
            cost_list = self.getDynamicCosts(stage_number, self.stages_definition.stage_dep_costs)
            for cost in cost_list:
                stage_coststack.addCost(*cost)

            stage_model = aligator.StageModel(stage_coststack, self.discrete_dynamics)
            for constraint in self.stages_definition.constraints:
                stage_model.addConstraint(*constraint)
            self.stage_model_list.append(stage_model)
        
        
        return stage_model

    def getTerminalCoststack(self, stage_number):
        """"
        Returns the terminal coststack calculated from the `stages_definitions` dict.
        """
        terminal_coststack = aligator.CostStack(self.space, self.nu)
        cost_list = self.getDynamicCosts(stage_number, self.stages_definition.terminal_costs)
        for cost in cost_list:
            terminal_coststack.addCost(*cost)
        #!! TODO: add terminal constraintStack

        # for terminal_cost in self.stages_definition.terminal_costs:
        #     terminal_coststack.addCost(*terminal_cost)

        return terminal_coststack

    def fabricateStages(self, current_stage, duration):
        """
        Builds the stages from `current_stage` to `duration` using the costs stored in `self.stages_definition`
        """

        terminal_coststack = aligator.CostStack(self.space, self.nu)
        for terminal_cost in self.stages_definition.terminal_costs:
            terminal_coststack.addCost(*terminal_cost)
        # terminal_coststack = 0
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
    # Cost & Constraints functions
    # ==========================================================================
    def addJointsLimitsConstraints(self) -> None:
        """
        Adds joints limits constraints (joint_angle_residual, box_constraint) to self.stages_definition["constraints"]
        """
        # Résidu de base qui extrait simplement l'état x.
        # C'est f(x) = x - 0, donc il renvoie x.
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

    def addTorqueLimitsConstraints(self):
        """
        Adds torque limits constraints (joint_angle_residual, box_constraint) to self.stages_definition["constraints"]
        """
        nv = self.robot.model.nv
        nu = nv
        ndx = self.space.ndx
        residual = aligator.ControlErrorResidual(ndx, nu)
        constraint = constraints.BoxConstraint(self.u_min, self.u_max)
        self.stages_definition.constraints.append((residual, constraint))

    def addRegulationCosts(self):
        ## Running costs
        # State reg
        wt_x = np.diag(
        [self.parameters.mpc.weights.running.regulation.joint] * self.nv
        +
        [self.parameters.mpc.weights.running.regulation.vel] * self.nv)

        position_ref = [-8.97653063991213e-07
        ,-0.7808463663675579
        ,2.2971344285441086e-15
        ,-2.366688685342141
        ,-9.726534605596857e-06
        ,1.5702636943341912
        ,0.7800000000000175]
        vel_ref = [0.]*7

        x_ref = np.array(position_ref + vel_ref)

        stage_reg_cost = [(f"reg_state_{i}", aligator.QuadraticStateCost(self.space, self.nu, x_ref, wt_x)) for i in range(self.parameters.mpc.n_total_steps)]

        if(self.parameters.mpc.weights.running.regulation.vel > 0. or self.parameters.mpc.weights.running.regulation.joint > 0.):
            self.stages_definition.stage_dep_costs.append(stage_reg_cost)
        
        # Control reg
        wt_u = self.parameters.mpc.weights.running.regulation.command*np.eye(self.nu)

        control_reg_cost = [(f"reg_ctrl_{i}", aligator.QuadraticControlCost(self.space, np.zeros(self.nu), wt_u)) for i in range(self.parameters.mpc.n_total_steps)]

        if(self.parameters.mpc.weights.running.regulation.command > 0.):
            self.stages_definition.stage_dep_costs.append(control_reg_cost)

        ## Terminal costs
        # State reg
        wt_x_term = self.parameters.mpc.weights.terminal.regulation.joint*np.ones(self.ndx)
        wt_x_term[self.nq:] = self.parameters.mpc.weights.terminal.regulation.vel
        wt_x_term = np.diag(wt_x_term)

        if(self.parameters.mpc.weights.terminal.regulation.vel > 0. or self.parameters.mpc.weights.terminal.regulation.joint > 0.):
            self.stages_definition.terminal_costs.append(("reg_state_term", aligator.QuadraticStateCost(self.space, self.nu, np.zeros(self.space.ndx), wt_x_term)))
        
        # Control reg
        wt_u_term = self.parameters.mpc.weights.terminal.regulation.command*np.eye(self.nu)

        if(self.parameters.mpc.weights.terminal.regulation.command > 0.):
            self.stages_definition.terminal_costs.append(("reg_ctrl_term", aligator.QuadraticControlCost(self.space, np.zeros(self.nu), wt_u_term)))


    def addWaypointCosts(self):
        """
        For each stage, adds a cost tied to matching the end effector frame to a waypoint frame
        """
        tool_id = self.robot.model.getFrameId(self.parameters.robot.tool_frame_name)
        frame_vel_cost = []
        placement_costs = []
        for t in range (self.parameters.mpc.n_total_steps):
            pose , target_vel = self.interpolator(t*self.parameters.mpc.dt)

            placement_residual = aligator.FramePlacementResidual(self.ndx, self.nu, self.robot.model, pose, self.robot.model.getFrameId(self.parameters.robot.tool_frame_name))

            wt_frame_pose = np.diag( [self.parameters.mpc.weights.running.waypoints.pose.translation]*3 + [self.parameters.mpc.weights.running.waypoints.pose.orientation]*3)
            cost = (f"pose_{t}", aligator.QuadraticResidualCost(self.space, placement_residual, wt_frame_pose))
            placement_costs.append(cost)
        
            # cost on the velocity of the waypoint
            frame_vel_fn = aligator.FrameVelocityResidual(self.ndx, self.nu, self.robot.model, target_vel, tool_id, pin.WORLD)
            wt_frame_vel = np.diag( [self.parameters.mpc.weights.running.waypoints.vel.translation]*3 + [self.parameters.mpc.weights.running.waypoints.vel.orientation]*3)
            cost_vel = (f"frame_vel_{t}", aligator.QuadraticResidualCost(self.space, frame_vel_fn, wt_frame_vel))
            frame_vel_cost.append(cost_vel)


        if(self.parameters.mpc.weights.running.waypoints.pose.translation > 0 or self.parameters.mpc.weights.running.waypoints.pose.orientation > 0):
            self.stages_definition.stage_dep_costs.append(placement_costs)
        if(self.parameters.mpc.weights.running.waypoints.vel.translation > 0 or self.parameters.mpc.weights.running.waypoints.vel.orientation > 0):
            self.stages_definition.stage_dep_costs.append(frame_vel_cost)


    def addOrientationCosts(self):
        """
        For each stage, adds a cost to align the end effector to the tangent of the trajectory
        """


    def addAutoCollisionsConstraints(self):
        """
        Adds a cost in `self.stages_definition["stage dependant costs"]` linked to self collisions of the robot (based on the collision pairs defined in the SRDF loaded by `example-robot-data`).
        WIP for now
        """

        #! test
        pairs_2_add = [23,35]
        for i in pairs_2_add:
            collision_residual = aligator.FrameCollisionResidual(self.ndx, self.nu, self.robot.model, self.robot.collision_model, i)
            # log barrier : weight*ln(function)
            self.stages_definition.stage_indep_costs.append((f"collision_{i}", aligator.LogResidualCost(self.space, collision_residual, self.parameters.mpc.weights.collision * np.eye(collision_residual.nr))))

            collision_constraint = constraints.BoxConstraint(np.array([0.05]), np.array([100]))
            self.stages_definition.constraints.append((collision_residual, collision_constraint))

    # ==========================================================================
    # Utils
    # ==========================================================================

    def getDynamicCosts(self, stage_number, master_cost_list):
        """
        Returns a list of costs for the current stage. If a cost is not defined for a given `stage_number`, the last instance of this cost is returned instead.
        """
        return_costs_list = []
        for cost_list in master_cost_list : #self.stages_definition.stage_dep_costs:
            if stage_number < len(cost_list):
                return_costs_list.append(cost_list[stage_number])
            else:
                return_costs_list.append(cost_list[-1])
        return return_costs_list

    def getStagesDefinition(self):
        return self.stages_definition

    def getStagesList(self): #! remove?
        return self.stages

    # def getFullTrajectory(self): # TODO : move to pattern generator?
    #     """
    #     Returns the full trajectory formatted as a np.array([np.array([x0,x1,..]), np.array([y0,y1,..]), np.array([z0,z1,..])])
    #     """
    #     target = self.spline.get_interpolated_pose(0)
    #     traj_x = np.array([float(target[0])])
    #     traj_y = np.array([float(target[1])])
    #     traj_z = np.array([float(target[2])])
    #     for i in range(self.n_steps):
    #         target = self.spline.get_interpolated_pose(i*self.parameters.mpc.dt)
    #         traj_x = np.append(traj_x, float(target[0]))
    #         traj_y = np.append(traj_y, float(target[1]))
    #         traj_z = np.append(traj_z, float(target[2]))
    #     traj = np.array([traj_x, traj_y, traj_z])
    #     return traj

    def getFullTrajectory_pt_by_pt(self): # TODO : move to pattern generator?
        """
        Return the full trajectory formatted as np.array([x0,y0,z0], [x1,y1,z1], ...)
        """
        traj = []
        for i in range(self.parameters.mpc.n_total_steps):
            pos, _ = self.interpolator(i*self.parameters.mpc.dt)
            traj.append(pos.translation)
        return traj
