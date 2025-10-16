import aligator
from aligator import constraints, manifolds, dynamics

import example_robot_data as ex_robot_data
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import hppfcl

from mpc_utils import SplineGenerator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Literal, List, Optional
import tap

import time
import sys

class ArgsBase(tap.Tap):
    display: bool = False  # Display the trajectory using meshcat
    plot: bool = False
class Args(ArgsBase):
    plot:bool = True
    fddp: bool = False
    bounds: bool = False
    collisions: bool = False
    debug : bool = False
    joints_limits : bool = False
    record : bool = False
    viz_traj : bool = False
    perturbate : bool = False
    orientation : bool = False

args = Args().parse_args()

print(args)

class MPC():
    def __init__(self):
        self.parameters = Params()
        print(self.parameters)

        self.robot = ex_robot_data.load(self.parameters.robot_name)
        self.space = self.space = manifolds.MultibodyPhaseSpace(self.robot.model)
        self.discrete_dynamics = self.calcDiscreteDynamics()

        self.n_dx = self.space.ndx # taille du vecteur d'état du robot
        self.n_q = self.robot.model.nq # taille du vecteur articulaire
        self.n_v = self.robot.model.nv # taille du vecteur vitesse
        self.nu = self.n_v
        # initialisation de la pose
        self.x0 = self.space.neutral()
        self.q0 = self.x0[:self.n_q] # état articulaire initial 0.0
        # changement d'état initial
        q0 = self.parameters.start_pose
        self.x0[:self.n_q] = q0
        self.tool_id = self.robot.model.getFrameId(self.parameters.tool_frame_name)
        # update model
        pin.forwardKinematics(self.robot.model, self.robot.data, q0)
        pin.updateFramePlacement(self.robot.model, self.robot.data, self.tool_id)

        self.stage_factory = GlueStageFactory(self.robot.model, self.space, self.parameters.n_total_steps, self.discrete_dynamics)

    def mpcLoop(self):
        nb_steps_mpc = self.parameters.mpc_steps
        solver, callback = self.instanciateSolver()

        final_results_xs = [self.x0] # add the start state
        final_results_us = []
        prim_infeas = [None]
        dual_infeas = [None]
        loop_times = []

        print("Starting MPC loop")
        for t in range (nb_steps_mpc):
            # if args.debug:
            print("t: "+ str(t))
            start = time.time()
            if t == 0:
                # first iteration
                stages, terminal_coststack = self.stage_factory.fabricateStages(t,self.parameters.mpc_steps)
                problem = aligator.TrajOptProblem(self.x0, stages, terminal_coststack)
                solver.setup(problem)

                # warm start
                us = [self.computeQuasistatic(self.robot.model, self.x0, a = np.zeros(self.n_v)) for _ in range(self.parameters.mpc_steps)]
                xs = aligator.rollout(self.discrete_dynamics, self.x0, us)

            else:
                stages, terminal_coststack = self.stage_factory.fabricateStages(t,self.parameters.mpc_steps)
                problem = aligator.TrajOptProblem(self.x0, stages, terminal_coststack)

                us = results.us.tolist()
                us = us[1:]
                us.append(us[-1])

                xs = results.xs.tolist()
                xs = xs[1:]
                xs.append(xs[-1])

                if args.perturbate:
                    xs[0] = np.add(xs[0], np.random.rand(18)*0.008) # pertubation on the state (max without exploding is ~0.01)

            # boucler avec aligator rollout
            results = self.run_solver(solver, problem, us, xs)
            stop = time.time()

            final_results_us.append(results.us.tolist()[0])
            final_results_xs.append(results.xs.tolist()[0])
            prim_infeas.append(callback.dual_infeas.tolist()[-1])
            dual_infeas.append(callback.prim_infeas.tolist()[-1])
            loop_times.append(stop-start)

        return final_results_us, final_results_xs, prim_infeas, dual_infeas, loop_times

    def computeQuasistatic(self, model: pin.Model, x0, a):
        data = model.createData()
        q0 = x0[:self.n_q]
        v0 = x0[self.n_q : self.n_q + self.n_v]

        return pin.rnea(model, data, q0, v0, a)

    def run_solver(self, solver, problem, us, xs):
        """
        Runs the solver over 'max_iters' iterations
        """
        start = time.time()
        solver.run(problem, xs, us)
        end = time.time()
        results = solver.results

        if args.debug:
            print("MPC calc time: " + str(end - start))
            print(results)
        return results

    def calcDiscreteDynamics(self):
        nu = self.robot.model.nv
        B_mat = np.eye(nu)
        ode = dynamics.MultibodyFreeFwdDynamics(self.space, B_mat) # Ordinatry Diff Equation: resolution de l'équation de la dynamique
        return dynamics.IntegratorSemiImplEuler(ode, self.parameters.dt)

    def instanciateSolver(self):
        solver = aligator.SolverProxDDP(self.parameters.solver_tolerance, self.parameters.mu_init, max_iters=self.parameters.mpc_max_iter, verbose=self.parameters.verbose)
        solver.rollout_type = self.parameters.solver_rollout_type
        solver.sa_strategy = self.parameters.solver_sa_strategy
        callback = aligator.HistoryCallback(solver)
        solver.registerCallback("his", callback)
        return solver, callback



class BaseStageFactory():
    def __init__(self, robot, space, n_steps, discrete_dynamics):
        self.robot = robot
        self.space = space
        self.nv = self.robot.model.nv
        self.nu = self.nv
        self.ndx = self.space.ndx
        self.discrete_dynamics = discrete_dynamics

        self.n_steps = n_steps

        self.parameters = Params()

        self.stages_definition: dict[str,list[tuple|list]] = {"constraints":[], "terminal costs":[], "stage dependant costs":[]} # Constraints are all global
        self.stages : list[aligator.stageModel] = []
        self.problem : aligator.TrajOptProblem = None

        # Add base costs & constraints present in all problems:
        self.addJointsLimitsConstraints()
        self.addTorqueLimitsConstraints()
        self.addRegulationCosts()

    def fabricateStages(self, current_stage, duration)->(list[aligator.StageModel], aligator.CostStack):
        """
        Builds the stages from `current_stage` to `duration`
        """
        terminal_coststack = aligator.CostStack(self.space, self.nu)
        for terminal_cost in self.stages_definition["terminal_costs"]:
            terminal_coststack.add(*terminal_cost)

        stages = []
        for i in range(current_stage, duration + current_stage):
            stage_coststack = aligator.CostStack(self.space, self.nu)
            cost = self.getDynamicCosts(current_stage)
            stage_coststack.add(*cost)
            stage_model = aligator.StageModel(stage_coststack, self.discrete_dynamics)
            for constraint in self.stages_definition["constraints"]:
                stage_model.add(*constraint)
            stages.append(stage_model)

        return stages, terminal_coststack

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

            # print(f"Adding BoxConstraint for Joint '{jn}': [{q_min:.3f}, {q_max:.3f}]")

            A = np.zeros((1, self.ndx))
            A[0, q_idx_in_x] = 1.0
            b = np.array([0.0])
            joint_angle_residual = aligator.LinearFunctionComposition(identity_residual, A, b)

            box_constraint = constraints.BoxConstraint(np.array([q_min]), np.array([q_max]))

            constraint = (joint_angle_residual, box_constraint)
            self.stages_definition["constraints"].append(constraint)

    def addTorqueLimitsConstraints(self):
        """
        Adds torque limits constraints (joint_angle_residual, box_constraint) to self.stages_definition["constraints"]
        """
        nv = self.robot.model.nv
        nu = nv
        ndx = self.space.ndx
        self.u_max = self.robot.model.effortLimit
        self.u_min = -self.u_max
        residual = aligator.ControlErrorResidual(ndx, nu)
        constraint = constraints.BoxConstraint(self.u_min, self.u_max)
        self.stages_definition["constraints"].append((residual, constraint))

    def addRegulationCosts(self):
        wt_x = self.parameters.joint_reg_cost*np.ones(self.ndx)
        wt_x[self.nv:] = self.parameters.vel_reg_cost
        wt_x = np.diag(wt_x)
        wt_u = self.parameters.command_reg_cost*np.eye(self.nu)

        # add Global target
        wt_x_term = wt_x.copy() # np.ones(self.n_dx)
        wt_x_term[:] = self.parameters.term_state_reg_cost

        terminal_cost = ("term reg", aligator.QuadraticCost(wt_x_term, wt_u * 0))
        self.stages_definition["terminal costs"].append(terminal_cost)

        stage_reg_cost = [("reg", aligator.QuadraticCost(wt_x * self.parameters.dt, wt_u * self.parameters.dt))]
        self.stages_definition["stage dependant costs"].append(stage_reg_cost)

    def addOrientationCosts(self):
        rpy = self.parameters.tool_orientation
        R = pin.rpy.rpyToMatrix(rpy)
        target_orientation = pin.Quaternion(R)

        target_placement = pin.SE3(target_orientation, np.zeros(3)) # on a juste besoin de la rotation

        placement_residual = aligator.FramePlacementResidual(self.ndx, self.nu, self.robot.model, target_placement, self.robot.model.getFrameId(self.parameters.tool_frame_name)) # [err_pos(3), err_ori(3)]

        # L'entrée est le vecteur 6D du placement_residual. La sortie doit être le vecteur 3D de l'erreur d'orientation.
        A_selector = np.hstack([np.zeros((3, 3)), np.eye(3)]) # sélectionne la partie rotation (les 3 dernières composantes)
        b_selector = np.zeros(3) # on veut que l'erreur soit nulle

        # Ce nouveau résidu ne sortira que la partie orientation de l'erreur de pose.
        orientation_only_residual = aligator.LinearFunctionComposition(placement_residual, A_selector, b_selector)

        cost = [("orientation", aligator.QuadraticResidualCost(self.space, orientation_only_residual, self.parameters.orientation_weight * np.eye(3)))]

        self.stages_definition["stage dependant costs"].append(cost)

    def getDynamicCosts(self, current_stage_num):
        """
        Returns a list of cost for the current stage. If a cost is not defined for a given t, the last instance of this cost is returned instead.
        """
        stage_costs = []
        for cost_list in self.stages_definition["stage dependant costs"]:
            if current_stage_num < len(cost_list):
                stage_costs.append(cost_list[current_stage_num])
            else:
                stage_costs.append(cost_list[-1])
        return stage_costs

    def addAutoCollisionsConstraints(self):
        #TODO
        pass

    def getStagesDefinition(self):
        return self.stages_definition

    def getStageModel(self):
        return self.stages

class GlueStageFactory(BaseStageFactory):
    def __init__(self, robot, space, n_steps, discrete_dynamics):
        super().__init__(robot, space, n_steps, discrete_dynamics)
        self.addWaypointCosts()
        self.addOrientationCosts()

        list_costs = self.getDynamicCosts(2)
        print("trunc costs " + str(list_costs))

    def addWaypointCosts(self):
        tool_id = self.robot.model.getFrameId(self.parameters.tool_frame_name)
        start_pos = self.robot.data.oMf[tool_id].translation.copy()
        spline = SplineGenerator(self.parameters.waypoints, start_pos ,self.parameters.total_time)
        tool_id = self.robot.model.getFrameId(self.parameters.tool_frame_name)
        waypoint_costs = []
        for t in range (self.n_steps):
            target_pos = spline.interpolate_pose(t) # TODO rajouter l'interpolation d'orientation
            frame_pos_fn = aligator.FrameTranslationResidual(self.ndx, self.nu, self.robot.model, target_pos, tool_id)
            v_ref = pin.Motion()
            v_ref.np[:] = 0
            frame_vel_fn = aligator.FrameVelocityResidual(self.ndx, self.nu, self.robot.model, v_ref, tool_id, pin.LOCAL)

            wt_x_term = np.zeros((self.ndx, self.ndx))
            wt_x_term[:] = self.parameters.waypoint_x_weight
            wt_frame_pos = self.parameters.waypoint_frame_pos_weight * np.eye(frame_pos_fn.nr)

            wt_frame_vel = self.parameters.waypoint_frame_vel_weight* np.ones(frame_vel_fn.nr)
            wt_frame_vel = np.diag(wt_frame_vel)

            cost = ("frame", aligator.QuadraticResidualCost(self.space, frame_pos_fn, wt_frame_pos))

            waypoint_costs.append(cost)
        self.stages_definition["stage dependant costs"].append(waypoint_costs)

# TODO class visu?

class Params():
    def __init__(self)->None:
        # Robot
        self.robot_name : str = "panda"
        self.world_frame_name : str = "universe"
        self.start_pose : np.ndarray = np.array([0, 0.0, 0.0, -1, 0.0, 2, 0.0, 0.0, 0.0])
        self.tool_frame_name : str = "panda_hand_tcp"

        # MPC
        self.dt : float = 0.01 # time is in seconds
        self.total_time : int|float = 0.05
        self.n_total_steps : int = int(self.total_time / self.dt)
        self.mpc_horizon : int|float = 1 # in seconds
        self.mpc_steps : int = int(self.mpc_horizon / self.dt)
        self.mpc_max_iter : int = 3
        self.solver_tolerance = 1e-7
        self.solver_rollout_type = aligator.ROLLOUT_NONLINEAR
        self.solver_sa_strategy = aligator.SA_LINESEARCH_NONMONOTONE
        self.mu_init = 1e-7
        self.verbose = aligator.VerboseLevel.QUIET #or VERBOSE

        # Weights:
        self.joint_reg_cost = 1e-2 #1e-4
        self.vel_reg_cost = 1e-2
        self.command_reg_cost = 1e-4
        self.term_state_reg_cost = 1e-4
        self.waypoint_x_weight = 1e-4
        self.waypoint_frame_pos_weight = 200.0
        self.waypoint_frame_vel_weight = 1
        self.orientation_weight = 50

        # Trajectory
        self.waypoints = [np.array([0.5, 0.0, 0.5]), np.array([0.5, 0.2, 0.5]), np.array([0.5, 0.2, 0.3]), np.array([0.5, 0.0, 0.3])] # ZY square
        self.tool_orientation = np.array([np.pi, 0., 0.])

    def __repr__(self)->str:
        """
        Formats the output when printing the object
        """
        return f'Robot parameters:\n'\
                    f'\tRobot name: {self.robot_name}\n'\
                    f'\tStart pose: {self.start_pose} (rads)\n'\
                    f'\tWorld frame name: "{self.world_frame_name}"\n'\
                    f'\tTool frame name: "{self.tool_frame_name}"\n'\
                f'\nMPC parameters:\n'\
                    f'\tdt: {self.dt} (secs)\n'\
                    f'\tTotal time: {self.total_time} (secs)\n'\
                    f'\tTotal number of steps: {self.n_total_steps}\n'\
                    f'\tHorizon: {self.mpc_horizon} (secs)\n'\
                    f'\tHorizon: {self.mpc_steps} (steps)\n'\
                    f'\tNumber max of iterations: {self.mpc_max_iter}\n'\
                f'\nWeights parameters:\n'\
                f'\tRegulations costs:\n'\
                    f'\t\tJoints: {self.joint_reg_cost}\n'\
                    f'\t\tVelocity: {self.vel_reg_cost}\n'\
                    f'\t\tCommand: {self.command_reg_cost}\n'\
                    f'\t\tTerminal state: {self.term_state_reg_cost}\n'\
                f'\n\tWaypoints:\n'\
                    f'\t\tState: {self.waypoint_x_weight}\n'\
                    f'\t\tFrame position: {self.waypoint_frame_pos_weight}\n'\
                    f'\t\tFrame velocity: {self.waypoint_frame_vel_weight}\n'\
                f'\n\tOrientation: {self.orientation_weight}\n'\



if __name__ == "__main__":
    mpc = MPC()
