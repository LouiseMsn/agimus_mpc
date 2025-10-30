from mpcTrajectoryUtils import SplineGenerator, PatternGenerator
from mpcParameters import Params, args

import aligator
from aligator import constraints, manifolds, dynamics
import example_robot_data as ex_robot_data
import pinocchio as pin
from pinocchio.visualize import ViserVisualizer
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List
import tap
import time
from copy import deepcopy


class MPC():
    def __init__(self, waypoints):
        print(args)
        self.parameters = Params()
        self.waypoints = waypoints
        print(self.parameters)

        # Initialize robot
        self.robot = ex_robot_data.load(self.parameters.robot_name)
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
        q0 = self.parameters.start_pose
        self.x0[:self.n_q] = q0
        pin.forwardKinematics(self.robot.model, self.robot.data, q0)
        pin.updateFramePlacement(self.robot.model, self.robot.data, self.tool_id) # update model placemement

        self.discrete_dynamics = self.calcDiscreteDynamics()
        self.stage_factory = GlueStageFactory(self.robot, self.space, self.parameters.n_total_steps, self.discrete_dynamics, waypoints)

        # Min & Max torque on command output
        self.u_min = self.stage_factory.u_min
        self.u_max = self.stage_factory.u_max
        pass

    def mpcLoop(self):
        """
        Simulates a MPC running for a total of `self.parameters.total_time` seconds
        """
        solver, callback = self.instanciateSolver()

        final_results_xs = [self.x0] # add the start state
        final_results_us = []
        prim_infeas = [None]
        dual_infeas = [None]
        loop_times = []


        for t in range (self.parameters.n_total_steps):
            if args.debug:
                print("t: "+ str(t)) #! debug
            start = time.time()
            if t == 0:
                # first iteration
                stages, terminal_coststack = self.stage_factory.fabricateStages(t, self.parameters.mpc_steps)
                # stages, terminal_coststack = self.stage_factory.hardcoded_stages(t, self.parameters.mpc_steps)
                problem = aligator.TrajOptProblem(self.x0, stages, terminal_coststack)
                solver.setup(problem)

                # warm start
                us = [self.computeQuasistatic(self.robot.model, self.x0, a = np.zeros(self.n_v)) for _ in range(self.parameters.mpc_steps)]
                xs = aligator.rollout(self.discrete_dynamics, self.x0, us)
                lams = []
                vs = []

            else:
                us = results.us.tolist()
                us = us[1:]
                us.append(us[-1])

                xs = results.xs.tolist()
                xs = xs[1:]
                xs.append(xs[-1])

                vs = results.vs.tolist()
                vs = vs[1:]
                vs.append(vs[-1])

                lams = results.lams.tolist()
                lams = lams[1:]
                lams.append(lams[-1])

                stages, terminal_coststack = self.stage_factory.fabricateStages(t,self.parameters.mpc_steps)

                # stages, terminal_coststack = self.stage_factory.hardcoded_stages(t, self.parameters.mpc_steps)
                problem = aligator.TrajOptProblem(xs[0], stages, terminal_coststack)

                if args.perturbate:
                    xs[0] = np.add(xs[0], np.random.rand(18)*0.008) # pertubation on the state (max without exploding is ~0.01)

            # boucler avec aligator rollout
            results = self.run_solver(solver, problem, us=us, xs=xs) #, lams=lams, vs=vs)

            stop = time.time()


            current_xs = deepcopy(results.xs.tolist()[0])
            current_us = deepcopy(results.us.tolist()[0])
            last_dual_infeas = deepcopy(callback.dual_infeas.tolist()[-1])
            last_prim_infeas = deepcopy(callback.prim_infeas.tolist()[-1])
            # print(f'current xs: {current_xs}')
            final_results_us.append(current_us)
            final_results_xs.append(current_xs)
            # if len(callback.dual_infeas.tolist()) > 0:
            prim_infeas.append(last_dual_infeas)
            dual_infeas.append(last_prim_infeas)
            loop_times.append(stop-start)

        return final_results_us, final_results_xs, prim_infeas, dual_infeas, loop_times

    def computeQuasistatic(self, model: pin.Model, x0, a):
        data = model.createData()
        q0 = x0[:self.n_q]
        v0 = x0[self.n_q : self.n_q + self.n_v]

        return pin.rnea(model, data, q0, v0, a)

    def run_solver(self, solver, problem, *, us, xs):#, lams, vs):
        """
        Runs the solver over 'max_iters' iterations
        """
        # solver.preg_ = 1e-09
        # solver.preg_init = 1e-09
        # solver.reg_init = 1e-09
        # solver.ls_params.alpha_min = 1


        # solver.reg_min
        # solver.x_reg


        start = time.time()
        solver.run(problem, xs, us) #, vs, lams) # TODO warm start vs and lams?
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
        self.u_max = self.robot.model.effortLimit
        self.u_min = -self.u_max
        self.discrete_dynamics = discrete_dynamics
        self.n_steps = n_steps

        self.parameters = Params()

        self.stages_definition: dict[str,list[tuple|list]] = {"constraints":[], "terminal costs":[], "stage dependant costs":[]} # Constraints are not stage-dependants
        self.stages : list[aligator.stageModel] = []
        self.problem : aligator.TrajOptProblem

        # Add base costs & constraints present in all problems:
        if not args.no_joints_lim:
            self._addJointsLimitsConstraints()
        if not args.no_torque_lim:
            self._addTorqueLimitsConstraints()
        self._addRegulationCosts()

    def fabricateStages(self, current_stage, duration):
        """
        Builds the stages from `current_stage` to `duration` using the costs stored in `self.stages_definition`
        """

        terminal_coststack = aligator.CostStack(self.space, self.nu)
        for terminal_cost in self.stages_definition["terminal costs"]:
        #     # print(f'Terminal cost: {terminal_cost}')
            terminal_coststack.addCost(*terminal_cost)

        # print("stages def:" + str(self.stages_definition))
        stages = []
        for stage_num in range(current_stage, duration + current_stage):
            stage_coststack = aligator.CostStack(self.space, self.nu)
            cost_list = self._getDynamicCosts(stage_num)
            # print(f'Waypoints costs list stage n°{stage_num}: {cost_list}')
            for cost in cost_list:
                stage_coststack.addCost(*cost)

            stage_model = aligator.StageModel(stage_coststack, self.discrete_dynamics)
            for constraint in self.stages_definition["constraints"]:
                stage_model.addConstraint(*constraint)
            stages.append(stage_model)

        return stages, terminal_coststack

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

            # print(f"Adding BoxConstraint for Joint '{jn}': [{q_min:.3f}, {q_max:.3f}]") #! debug

            A = np.zeros((1, self.ndx))
            A[0, q_idx_in_x] = 1.0
            b = np.array([0.0])
            joint_angle_residual = aligator.LinearFunctionComposition(identity_residual, A, b)

            box_constraint = constraints.BoxConstraint(np.array([q_min]), np.array([q_max]))

            constraint = (joint_angle_residual, box_constraint)
            self.stages_definition["constraints"].append(constraint)

    def _addTorqueLimitsConstraints(self):
        """
        Adds torque limits constraints (joint_angle_residual, box_constraint) to self.stages_definition["constraints"]
        """
        nv = self.robot.model.nv
        nu = nv
        ndx = self.space.ndx
        residual = aligator.ControlErrorResidual(ndx, nu)
        constraint = constraints.BoxConstraint(self.u_min, self.u_max)
        self.stages_definition["constraints"].append((residual, constraint))

    def _addRegulationCosts(self):
        wt_x = self.parameters.stage_joint_reg_cost*np.ones(self.ndx)
        wt_x[self.nv:] = self.parameters.stage_vel_reg_cost
        wt_x = np.diag(wt_x)
        wt_u = self.parameters.command_reg_cost*np.eye(self.nu)

        # add Global target
        wt_x_term = self.parameters.term_state_reg_cost*np.eye(self.ndx)

        terminal_cost = ("term reg", aligator.QuadraticCost(wt_x_term, wt_u * 0))
        self.stages_definition["terminal costs"].append(terminal_cost)

        stage_reg_cost = [("reg", aligator.QuadraticCost(wt_x * self.parameters.dt, wt_u * self.parameters.dt))]
        self.stages_definition["stage dependant costs"].append(stage_reg_cost)

        # todo decouple v 1e-3 q 1e-7

    def _getDynamicCosts(self, current_stage_num):
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

    def _addAutoCollisionsConstraints(self):
        #TODO
        pass

    def getStagesDefinition(self):
        return self.stages_definition

    def getStageModel(self):
        return self.stages

class GlueStageFactory(BaseStageFactory):
    def __init__(self, robot, space, n_steps, discrete_dynamics, waypoints):
        super().__init__(robot, space, n_steps, discrete_dynamics)
        self.waypoints = waypoints
        tool_id = self.robot.model.getFrameId(self.parameters.tool_frame_name)
        start_pos = self.robot.data.oMf[tool_id].translation.copy()

        start_ori = self.robot.data.oMf[tool_id].rotation.copy()
        start_ori_rpy = pin.rpy.matrixToRpy(start_ori)
        print(start_ori_rpy)

        self.spline = SplineGenerator(start_pos, start_ori_rpy, self.waypoints,v_spread=self.parameters.vel_spread, v_start=self.parameters.vel_start)
        if not args.no_waypoints:
            self._addWaypointCosts()

        if not args.no_orientation_cost:
            self._addOrientationCosts()


    def _addWaypointCosts(self):
        """
        For each stage, adds a cost tied to matching the end effector frame to a waypoint frame
        """
        tool_id = self.robot.model.getFrameId(self.parameters.tool_frame_name)
        waypoint_costs = []
        for t in range (self.parameters.n_total_steps):
            target_pos = self.spline.get_interpolated_pose(t*self.parameters.dt)
            frame_pos_fn = aligator.FrameTranslationResidual(self.ndx, self.nu, self.robot.model, target_pos, tool_id)
            v_ref = pin.Motion()
            v_ref.np[:] = 0

            wt_x_term = np.zeros((self.ndx, self.ndx))
            wt_x_term[:] = self.parameters.waypoint_x_weight
            wt_frame_pos = self.parameters.waypoint_frame_pos_weight * np.eye(frame_pos_fn.nr)

            cost = ("frame", aligator.QuadraticResidualCost(self.space, frame_pos_fn, wt_frame_pos))

            waypoint_costs.append(cost)
        self.stages_definition["stage dependant costs"].append(waypoint_costs)

    def _addOrientationCosts(self):
        """
        For each stage, adds a cost to align the end effector to the tangent of the trajectory
        """
        orientation_costs = []
        for t in range (self.parameters.n_total_steps):
            rpy = self.spline.get_interpolated_ori(t*self.parameters.dt)
            # print(f'rpy: {rpy}')
            R = pin.rpy.rpyToMatrix(rpy)
            target_orientation = pin.Quaternion(R)

            target_placement = pin.SE3(target_orientation, np.zeros(3)) # on a juste besoin de la rotation

            placement_residual = aligator.FramePlacementResidual(self.ndx, self.nu, self.robot.model, target_placement, self.robot.model.getFrameId(self.parameters.tool_frame_name)) # [err_pos(3), err_ori(3)]

            # L'entrée est le vecteur 6D du placement_residual. La sortie doit être le vecteur 3D de l'erreur d'orientation.
            A_selector = np.hstack([np.zeros((3, 3)), np.eye(3)]) # sélectionne la partie rotation (les 3 dernières composantes)
            b_selector = np.zeros(3) # on veut que l'erreur soit nulle

            # Ce nouveau résidu ne sortira que la partie orientation de l'erreur de pose.
            orientation_only_residual = aligator.LinearFunctionComposition(placement_residual, A_selector, b_selector)

            cost = ("orientation", aligator.QuadraticResidualCost(self.space, orientation_only_residual, self.parameters.orientation_weight * np.eye(3)))
            orientation_costs.append(cost)
        self.stages_definition["stage dependant costs"].append(orientation_costs)

    def getFullTrajectory(self):
        target = self.spline.get_interpolated_pose(0)
        traj_x = np.array([float(target[0])])
        traj_y = np.array([float(target[1])])
        traj_z = np.array([float(target[2])])
        for i in range(self.n_steps):
            target = self.spline.get_interpolated_pose(i*self.parameters.dt)
            traj_x = np.append(traj_x, float(target[0]))
            traj_y = np.append(traj_y, float(target[1]))
            traj_z = np.append(traj_z, float(target[2]))
        traj = np.array([traj_x, traj_y, traj_z])

        return traj

    def getFullTrajectory_pt_by_pt(self):
        traj = []
        for i in range(self.n_steps):
            target = self.spline.get_interpolated_pose(i*self.parameters.dt)
            traj.append(target)
        return traj


class Visualization():
    """
    Class used to visualize the results of a MPC run
    """
    def __init__(self,  mpc : MPC):
        self.mpc = mpc
        self.robot = self.mpc.robot
        self._instanciateVizer()

    def _instanciateVizer(self):
        """
        Instanciates the viewer and display the init position
        """
        self.vizer = ViserVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model, data=self.robot.data)
        self.vizer.initViewer(open=False, loadModel=True)
        self.vizer.viewer.scene.add_grid(
                                            "/grid",
                                            width=20.0,
                                            height=20.0,
                                            position=np.array([0.0, 0.0, 0]),
                                            )

        self.vizer.display(self.mpc.q0)
        time.sleep(10)

    def display(self, xs,):
        """
        Displays the traj in meshcat as well as graphs #TODO splits the graphs to separate function
        """
        waypoints = self.mpc.waypoints
        xs_opt = xs
        xs = np.array(xs)
        qs = xs[:,:self.mpc.n_q]
        pts = self.get_endpoint_traj(xs_opt)

        traj_executed = []
        for i in range(pts.T.shape[1]):
            traj_executed.append(np.array([float(pts.T[0][i]), float(pts.T[1][i]),float(pts.T[2][i])]))



        self.vizer.viewer.scene.add_spline_catmull_rom(
                                                        "Output traj",
                                                        points=traj_executed,
                                                        tension=0.5,
                                                        line_width=3.0,
                                                        color=np.array([6, 117, 255]),
                                                        segments=100,
                                                        )
        self.vizer.viewer.scene.add_spline_catmull_rom(
                                                        "Input traj",
                                                        points=self.mpc.stage_factory.getFullTrajectory_pt_by_pt(),
                                                        tension=0.5,
                                                        line_width=2.0,
                                                        color=np.array([255, 105, 105]),
                                                        segments=100,
        )


        input("[Press enter]")
        num_repeat = 10

        qs = [x[:self.mpc.n_q] for x in xs_opt]

        for i in range(num_repeat):

            start = time.time()
            self.vizer.play(qs, self.mpc.parameters.dt)
            stop = time.time()
            print("Playing time: " + str(stop - start))

            time.sleep(3)

    def plotResults(self, xs, us, prim_infeas, dual_infeas, mpc_loop_times):
        waypoints = self.mpc.waypoints
        xs_opt = xs
        us_opt = np.asarray(us)
        xs = np.array(xs)
        qs = xs[:,:self.mpc.n_q]
        pts = self.get_endpoint_traj(xs_opt)
        times = np.linspace(0.0, self.mpc.parameters.total_time , self.mpc.parameters.n_total_steps + 1 )

        fig: plt.Figure = plt.figure(constrained_layout=True)
        fig.set_size_inches(6.4, 6.4)

        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 2])
        _u_ncol = 2
        _u_nrow, rmdr = divmod(self.mpc.nu, _u_ncol)
        if rmdr > 0:
            _u_nrow += 1
        gs1 = gs[1, :].subgridspec(_u_nrow, _u_ncol)

        plt.subplot(gs[0, 0])
        plt.plot(times, xs_opt)
        plt.title("States")

        axarr = gs1.subplots(sharex=True)
        handles_ = []

        for i in range(self.mpc.nu):
            ax: plt.Axes = axarr.flat[i]
            ax.plot(times[1:], us_opt[:, i])
            hl = ax.hlines(
                (self.mpc.stage_factory.u_min[i], self.mpc.stage_factory.u_max[i]), *times[[0, -1]], linestyles="--", colors="r"
            )
            handles_.append(hl)
            fontsize = 7
            ax.set_ylabel("$u_{{%d}}$" % (i + 1), fontsize=fontsize)
            ax.tick_params(axis="both", labelsize=fontsize)
            ax.tick_params(axis="y", rotation=90)
            if i + 1 == self.mpc.nu - 1:
                ax.set_xlabel("time", loc="left", fontsize=fontsize)

        ax = plt.subplot(gs[0, 1], projection="3d")
        ax.plot(*pts.T, "r", lw=1.0, label="Actual traj") # actual traj
        ax.plot(*self.mpc.stage_factory.getFullTrajectory(), "g", lw=1, alpha=0.7, label="Command traj") # command trajectory

        for waypoint in self.mpc.waypoints:
            ax.scatter(*waypoint, marker="^", c="b",alpha=0.5,s=10)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        ax.set(xlim=(0, 0.5), ylim=(-0.5, 0.5), zlim=(0, 0.5))
        ax.legend()

        # Primal and dual error plot
        plt.figure(2)
        plt.subplot(2,1,1)
        ax: plt.Axes = plt.gca()
        plt.plot(times, prim_infeas, ls="--", marker=".", label="primal error")
        plt.plot(times, dual_infeas, ls="--", marker=".", label="dual error")
        ax.set_xlabel("Stage number")
        ax.set_yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.subplot(212)
        ax: plt.Axes = plt.gca()
        plt.plot(times[1:], mpc_loop_times, marker=".", color="dodgerblue", label="Calculation time ")
        ax.set_xlabel("MPC loop time for t")
        ax.set_ylabel("Calculation time (secs)")


        # joints limit plot
        fig, ax = plt.subplots(4,2,sharex=True)

        ax = ax.flat
        current_ax = next(ax)
        for j_idx, jn in enumerate(self.robot.model.names[1:8]):
            q_idx_in_x = self.robot.model.joints[j_idx + 1].idx_q
            q_min = self.robot.model.lowerPositionLimit[q_idx_in_x]
            q_max = self.robot.model.upperPositionLimit[q_idx_in_x]
            current_ax.plot(times, qs[:, q_idx_in_x])

            current_ax.fill_between(times,q_min, q_max,alpha=0.1, color="mediumspringgreen")
            current_ax.set_ylabel("${{%s}} [id:{{%d}}]  (radiants)$" % (jn ,j_idx + 1), fontsize=fontsize)
            current_ax = next(ax)

        plt.show()

    def get_endpoint_traj(self, xs: List[np.ndarray]):
        """
        Gets the trajectory of the effector for a state list
        """
        pts = []
        for i in range(len(xs)):
            pts.append(self.get_endpoint(xs[i][: self.mpc.n_q]))
        return np.array(pts)

    def get_endpoint(self, q: np.ndarray):
        """
        Gets the effector pose for a joint configuration q
        """
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
        return self.robot.data.oMf[self.mpc.tool_id].translation.copy()


if __name__ == "__main__":
    patternGen = PatternGenerator([0.5,0.5,0], (0.5,0,0))
    x,y,z = patternGen.generate_pattern('zigzag_curve',stride=0.1)
    positions :list = []
    for i in range (len(x)):
        positions.append(np.array([x[i], y[i], z[i]]))

 # ! debug ======================================================================
    # print(f'Positions: \n{positions}')
    # print("num de positions:" + str(len(positions)))

    # positions = [np.array([0.5, 0.0, 0.2]),
    #             np.array([ 0.5, 0.0, 0.5])]
                # ,
    #             np.array([0.35, 0.35, 0.5]),
    #             np.array([0.35, 0.35, 0.2]),
    #             np.array([0.0, 0.5, 0.2]),
    #             np.array([0.0, 0.5, 0.5]),
    #             np.array([-0.35, 0.35, 0.5]),
    #             np.array([-0.35, 0.35, 0.2]),
    #             np.array([-0.5, 0.0, 0.2]),
    #             np.array([-0.5, 0.0, 0.5]),
    #             np.array([-0.35, -0.35, 0.5]),
    #             np.array([-0.35, -0.35, 0.2]),
    #             np.array([0.0, -0.5, 0.2]),
    #             np.array([0.0, -0.5, 0.5]),
    #             np.array([0.35, -0.35,  0.5]),
    #             np.array([0.35, -0.35,  0.2]),
    #             np.array([0.5, 0.0, 0.2])]
# ! ============================================================================

    # print(positions)

    mpc = MPC(positions)
    final_results_us, final_results_xs, prim_infeas, dual_infeas, mpc_loop_times = mpc.mpcLoop()
    # final_results_us, final_results_xs, prim_infeas, dual_infeas, mpc_loop_times = mpc.loneRun()
    viz = Visualization(mpc)
    viz.plotResults(final_results_xs, final_results_us, prim_infeas, dual_infeas, mpc_loop_times)
    if args.viz3D:
        viz.display(final_results_xs)
