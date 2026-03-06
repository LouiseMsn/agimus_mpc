
import aligator
from aligator_mpc.StagesFactory import StagesFactory
from aligator_mpc.mpcUtils import getIndexesFromJointNames
from aligator_mpc.mpcParameters import Config
from aligator import constraints, manifolds, dynamics

import pinocchio as pin
import numpy as np
from typing import List
import time

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

        self.space = manifolds.MultibodyPhaseSpace(self.robot.model)
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

        self.stage_factory = StagesFactory(self.robot, self.space, self.parameters.task.mpc.n_total_steps, self.discrete_dynamics, self.waypoints, self.parameters)

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
