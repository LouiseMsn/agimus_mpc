from aligator_mpc.mpcTrajectoryUtils import Interpolator
from aligator_mpc.mpcParameters import args

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


# !! TEMP
from line_profiler import profile
from rclpy.impl import rcutils_logger

class MPC():
    def __init__(self, waypoints, parameters):
        print(args)
        self.parameters = parameters
        self.waypoints = waypoints
        print(self.parameters)
        # Initialize robot
        self.robot = ex_robot_data.load(self.parameters.robot_name)
        self.robot.model = pin.buildReducedModel(self.robot.model, [8,9], self.parameters.start_pose )
        self.robot.data = self.robot.model.createData()

        self.space = self.space = manifolds.MultibodyPhaseSpace(self.robot.model)
        self.tool_id = self.robot.model.getFrameId(self.parameters.tool_frame_name)
        self.world_frame_id = self.robot.model.getFrameId(self.parameters.world_frame_name)
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
        solver = aligator.SolverProxDDP(self.parameters.solver_tolerance, self.parameters.solver_mu_init, max_iters=self.parameters.mpc_max_iters, verbose=aligator.VerboseLevel.QUIET)
        solver.rollout_type = self.parameters.solver_rollout_type
        solver.sa_strategy = self.parameters.solver_sa_strategy
        solver.linear_solver_choice = self.parameters.solver_linear_solver_choice
        if self.parameters.solver_linear_solver_choice == aligator.LQ_SOLVER_PARALLEL:
            solver.setNumThreads(self.parameters.solver_num_threads)
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
        self.stage_factory = StageFactory(self.robot, self.space, self.parameters.n_total_steps, self.discrete_dynamics, self.waypoints, self.parameters)

        # Min & Max torque on command output
        self.u_min = self.stage_factory.u_min
        self.u_max = self.stage_factory.u_max

    def iterate(self, current_xs, current_us):
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
            us = [self.computeQuasistatic(self.robot.model, self.x0, a = np.zeros(self.n_v)) for _ in range(self.parameters.nb_steps_horizon)]
            xs = aligator.rollout(self.discrete_dynamics, self.x0, us)

            # create the stages & problem
            stages, terminal_coststack = self.stage_factory.fabricateStages(0, self.parameters.nb_steps_horizon)
            self.problem = aligator.TrajOptProblem(self.x0, stages, terminal_coststack)
            self.solver.setup(self.problem)
            self.solver.max_iters = self.parameters.mpc_max_iters_1st_iter
            self.solver.mu_init = self.parameters.solver_mu_init_1st_iter
            
        else:
            self.solver.max_iters = self.parameters.mpc_max_iters
            self.solver.mu_init = self.parameters.solver_mu_init
            # cycle the data
            us   = self.cycleData(self.results.us.tolist(), current_us, "xs")
            xs   = self.cycleData(self.results.xs.tolist(), current_xs,"xs")

            self.solver_stage_number = self.solver_stage_number 

            end_of_horizon_index = self.solver_stage_number + self.parameters.nb_steps_horizon - 1 # -1 because the first stage is 0

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

        if args.debug:
            print("MPC calc time: " + str(timer))
            print(results)
        return results, timer

    def calcDiscreteDynamics(self):
        """
        Initializes the discrete dynamic of the system.
        """
        nu = self.robot.model.nv
        B_mat = np.eye(nu)
        ode = dynamics.MultibodyFreeFwdDynamics(self.space, B_mat) # Ordinatry Diff Equation: resolution de l'équation de la dynamique
        return dynamics.IntegratorSemiImplEuler(ode, self.parameters.dt)

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

        self._addWaypointCosts()
        self._addOrientationCosts()
        self._addJointsLimitsConstraints()
        self._addTorqueLimitsConstraints()
        self._addRegulationCosts()
        # self._addAutoCollisionsConstraints()
        self.buildStageModelList()

    def getInterpolator(self):
        tool_id = self.robot.model.getFrameId(self.parameters.tool_frame_name)
        start_pos = self.robot.data.oMf[tool_id]        
        self.waypoints = [start_pos] + self.waypoints
        rcutils_logger.RcutilsLogger(name="   MPC_DEBUG   ").info(f'start {pin.rpy.matrixToRpy(start_pos.rotation)} { start_pos.translation}')
        return Interpolator(self.waypoints, self.parameters.vel_spread)

    def getStageModel(self, stage_number):
        """Returns the stage model for the _stage_number_ th stage

        Args:
            stage_number (int): _description_
        """
        if len(self.stage_model_list ) == 0:
            raise ValueError("The stage model list is not built, run buildStageModelList() before running getStageModel()")
        else:
            if stage_number >= self.parameters.n_total_steps :
                # rcutils_logger.RcutilsLogger(name="   MPC_DEBUG   ").info('fin de trajectoire')
                return self.stage_model_list[-1]
            else:
                return self.stage_model_list[stage_number]
        
    def buildStageModelList(self):
        """
        Builds and returns the StageModel for the `stage_number` th stage of the problem.
        """
        for stage_number in range(self.parameters.n_total_steps):
        
            stage_coststack = aligator.CostStack(self.space, self.nu)
            cost_list = self._getDynamicCosts(stage_number)
            for cost in cost_list:
                stage_coststack.addCost(*cost)

            stage_model = aligator.StageModel(stage_coststack, self.discrete_dynamics)
            for constraint in self.stages_definition.constraints:
                stage_model.addConstraint(*constraint)
            self.stage_model_list.append(stage_model)
        
        return stage_model

    def getTerminalCoststack(self):
        """"
        Returns the terminal coststack calculated from the `stages_definitions` dict.
        """
        terminal_coststack = aligator.CostStack(self.space, self.nu)
        for terminal_cost in self.stages_definition.terminal_costs:
            terminal_coststack.addCost(*terminal_cost)

        return terminal_coststack

    def fabricateStages(self, current_stage, duration):
        """
        Builds the stages from `current_stage` to `duration` using the costs stored in `self.stages_definition`
        """

        terminal_coststack = aligator.CostStack(self.space, self.nu)
        for terminal_cost in self.stages_definition.terminal_costs:
            terminal_coststack.addCost(*terminal_cost)
        stages = []
        for stage_num in range(current_stage, duration + current_stage):
            stage_coststack = aligator.CostStack(self.space, self.nu)
            cost_list = self._getDynamicCosts(stage_num)
            if len(self.stages_definition.stage_indep_costs) != 0:
                cost_list = cost_list + self.stages_definition.stage_indep_costs

            for cost in cost_list:
                stage_coststack.addCost(*cost)

            stage_model = aligator.StageModel(stage_coststack, self.discrete_dynamics)
            for constraint in self.stages_definition.constraints:
                stage_model.addConstraint(*constraint)
            stages.append(stage_model)

        return stages, terminal_coststack

    # ==========================================================================
    # Cost & Constraints functions
    # ==========================================================================
    def _addJointsLimitsConstraints(self) -> None:
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

    def _addTorqueLimitsConstraints(self):
        """
        Adds torque limits constraints (joint_angle_residual, box_constraint) to self.stages_definition["constraints"]
        """
        nv = self.robot.model.nv
        nu = nv
        ndx = self.space.ndx
        residual = aligator.ControlErrorResidual(ndx, nu)
        constraint = constraints.BoxConstraint(self.u_min, self.u_max)
        self.stages_definition.constraints.append((residual, constraint))

    def _addRegulationCosts(self):
        wt_x = self.parameters.stage_joint_reg_cost*np.ones(self.ndx)
        wt_x[self.nv:] = self.parameters.stage_vel_reg_cost
        wt_x = np.diag(wt_x)
        wt_u = self.parameters.command_reg_cost*np.eye(self.nu)

        # add Global target
        wt_x_term = self.parameters.term_state_reg_cost*np.eye(self.ndx)

        terminal_cost = ("term reg", aligator.QuadraticCost(wt_x_term, wt_u * 0))
        self.stages_definition.terminal_costs.append(terminal_cost)

        stage_reg_cost = [("reg", aligator.QuadraticCost(wt_x * self.parameters.dt, wt_u * self.parameters.dt))]
        self.stages_definition.stage_dep_costs.append(stage_reg_cost)

    def _addWaypointCosts(self):
        """
        For each stage, adds a cost tied to matching the end effector frame to a waypoint frame
        """
        tool_id = self.robot.model.getFrameId(self.parameters.tool_frame_name)
        waypoint_costs = []
        for t in range (self.parameters.n_total_steps):
            target_pos = self.interpolator(t*self.parameters.dt).translation

            frame_pos_fn = aligator.FrameTranslationResidual(self.ndx, self.nu, self.robot.model, target_pos, tool_id)
            v_ref = pin.Motion()
            v_ref.np[:] = 0

            wt_frame_pos = self.parameters.waypoint_frame_pos_weight * np.eye(frame_pos_fn.nr)

            cost = (f"frame_{t}", aligator.QuadraticResidualCost(self.space, frame_pos_fn, wt_frame_pos))

            waypoint_costs.append(cost)
        self.stages_definition.stage_dep_costs.append(waypoint_costs)

    def _addOrientationCosts(self):
        """
        For each stage, adds a cost to align the end effector to the tangent of the trajectory
        """
        orientation_costs = []
        for t in range (self.parameters.n_total_steps):
            R = self.interpolator(t*self.parameters.dt).rotation
            # R = pin.rpy.rpyToMatrix(rpy)
            target_orientation = pin.Quaternion(R)

            target_placement = pin.SE3(target_orientation, np.zeros(3)) # only take orientation

            placement_residual = aligator.FramePlacementResidual(self.ndx, self.nu, self.robot.model, target_placement, self.robot.model.getFrameId(self.parameters.tool_frame_name)) # [err_pos(3), err_ori(3)]

            # L'entrée est le vecteur 6D du placement_residual. La sortie doit être le vecteur 3D de l'erreur d'orientation.
            A_selector = np.hstack([np.zeros((3, 3)), np.eye(3)]) # sélectionne la partie rotation (les 3 dernières composantes)
            b_selector = np.zeros(3) # on veut que l'erreur soit nulle

            # Ce nouveau résidu ne sortira que la partie orientation de l'erreur de pose.
            orientation_only_residual = aligator.LinearFunctionComposition(placement_residual, A_selector, b_selector)

            cost = (f"orientation_{t}", aligator.QuadraticResidualCost(self.space, orientation_only_residual, self.parameters.orientation_weight * np.eye(3)))
            orientation_costs.append(cost)
        self.stages_definition.stage_dep_costs.append(orientation_costs)

    def _addAutoCollisionsConstraints(self):
        """
        Adds a cost in `self.stages_definition["stage dependant costs"]` linked to self collisions of the robot (based on the collision pairs defined in the SRDF loaded by `example-robot-data`).
        WIP for now
        """

        #! test
        pairs_2_add = [23,35]
        for i in pairs_2_add:
            collision_residual = aligator.FrameCollisionResidual(self.ndx, self.nu, self.robot.model, self.robot.collision_model, i)
            # log barrier : weight*ln(function)
            self.stages_definition.stage_indep_costs.append((f"collision_{i}", aligator.LogResidualCost(self.space, collision_residual, self.parameters.collision_weight * np.eye(collision_residual.nr))))

            collision_constraint = constraints.BoxConstraint(np.array([0.05]), np.array([100]))
            self.stages_definition.constraints.append((collision_residual, collision_constraint))

    # ==========================================================================
    # Utils
    # ==========================================================================

    def _getDynamicCosts(self, stage_number):
        """
        Returns a list of costs for the current stage. If a cost is not defined for a given `stage_number`, the last instance of this cost is returned instead.
        """
        stage_costs = []
        for cost_list in self.stages_definition.stage_dep_costs:
            if stage_number < len(cost_list):
                stage_costs.append(cost_list[stage_number])
            else:
                stage_costs.append(cost_list[-1])
        return stage_costs

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
    #         target = self.spline.get_interpolated_pose(i*self.parameters.dt)
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
        for i in range(self.parameters.n_total_steps):
            target = self.interpolator(i*self.parameters.dt)
            traj.append(target.translation)
        return traj
