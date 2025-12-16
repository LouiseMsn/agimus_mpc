import viser
import viser.uplot
import pinocchio as pin
from pinocchio.visualize import ViserVisualizer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from aligator_mpc.mpc import MPC

from typing import List

class Visualization():
    """
    Class used to visualize the results of a MPC run
    """
    def __init__(self,  mpc : MPC):
        self.mpc = mpc
        self.robot = self.mpc.robot
        self.client_connected = False

        # plot
        self.uplot_handles: list[viser.GuiUplotHandle] = []
        self.plot_time = np.array([0])
        self.plot_qs = [np.zeros(1) for i in range(self.mpc.n_q)]
        self._instantiateVizer()

    def _instantiateVizer(self):
        """
        Instanciates the viewer and display the init position
        """
        self.vizer = ViserVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model, data=self.robot.data)
        self.vizer.initViewer(open=False, loadModel=True)

        # Block until client connects
        self.vizer.viewer.on_client_connect(self._callbackClientConnect)
        while not self.client_connected :
            pass

        self.vizer.viewer.scene.add_grid(
                                        "/grid",
                                        width=20.0,
                                        height=20.0,
                                        position=np.array([0.0, 0.0, 0]),
                                        )

        self.vizer.viewer.scene.add_spline_catmull_rom(
                                                "Input traj",
                                                points=self.mpc.stage_factory.getFullTrajectory_pt_by_pt(),
                                                tension=0.5,
                                                line_width=2.0,
                                                color=np.array([255, 105, 105]),
                                                segments=100,
        )

        # add live plotting for the joint states
        self.uplot_handles: list[viser.GuiUplotHandle] = []
        for i in range(len(self.plot_qs)):
            data = (self.plot_time, self.plot_qs[i])
            self.uplot_handles.append(
                self.vizer.viewer.gui.add_uplot(
                    data=data,
                    series=(
                        viser.uplot.Series(label="time"),
                        *[
                            viser.uplot.Series(
                                label=f"q_{i}",
                                stroke="blue",
                                width=2,
                            )
                        ],
                    ),
                    title=f"Joint state for q_{i}",
                    scales={
                        "x": viser.uplot.Scale(
                            time=False,
                            auto=True,
                        ),
                        "y": viser.uplot.Scale(range=(-4,4)),
                    },
                    legend=viser.uplot.Legend(show=True),
                    aspect=2.0,
                )
            )
        self.vizer.display(self.mpc.q0)

    def _callbackClientConnect(self,_):
        """
        Called when a client connects and updates the `self.client_connected` to `True`
        """
        self.client_connected = True
        return None

    def _callbackVisualization(self,i):
        """
        Callback for each visualization step. Adds the input trajectory (horizon) viewed by the mpc at i step
        """
        t = self.mpc.solver_stage_number
        input_traj = self.mpc.stage_factory.getFullTrajectory_pt_by_pt()
        horizon_len = self.mpc.parameters.nb_steps_horizon

        if (t+horizon_len) > len(input_traj):
            horizon = input_traj[t:]
            for k in range(len(horizon), horizon_len):
                horizon.append(input_traj[-1])
        else:
            horizon = input_traj[t:t+self.mpc.parameters.nb_steps_horizon]

        self.vizer.viewer.scene.add_spline_catmull_rom(
                                                        "Horizon",
                                                        points=horizon,
                                                        tension=0.5,
                                                        line_width=4.0,
                                                        color=np.array([0, 255, 128]),
                                                        segments=100,
        )

    def update_plot(self,qs,t): # todo
        """
        Updates the joint limits plots
        """
        self.plot_time = np.append(self.plot_time, np.array([t*self.mpc.parameters.dt]))

        for i in range(len(self.plot_qs)):
            self.plot_qs[i] = np.append(self.plot_qs[i], np.array([qs[i]]))
            self.uplot_handles[i].data = (self.plot_time, self.plot_qs[i])

    def display_step(self, xs):
        """
        Displays only the first step of the `xs` passed as argument. Also updates the joint limits plot
        """
        xs_opt = xs
        xs = np.array(xs[0])
        qs = xs[:self.mpc.n_q] # take only the first xs and only the q part of it
        pts = self.get_endpoint_traj(xs_opt)

        future_trajectory = []
        for i in range(pts.T.shape[1]-1):
            future_trajectory.append(np.array([float(pts.T[0][i]), float(pts.T[1][i]),float(pts.T[2][i])]))

        self.vizer.viewer.scene.add_spline_catmull_rom(
                                            "Output traj",
                                            points=future_trajectory,
                                            tension=0.5,
                                            line_width=2.0,
                                            color=np.array([6, 117, 255]),
                                            segments=100,
                                            )
        self.vizer.play([qs], self.mpc.parameters.dt, callback=self._callbackVisualization)

    def display(self, xs): #? Depreciate?
        """
        Displays the traj in meshcat as well as graphs
        """

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
                                                        line_width=2.0,
                                                        color=np.array([6, 117, 255]),
                                                        segments=100,
                                                        )

        qs = [x[:self.mpc.n_q] for x in xs_opt]
        input_return = input("[Press enter to play, type \"q\" to exit]\n")
        while "q" not in input_return :
            self.vizer.play(qs, self.mpc.parameters.dt, callback=self._callbackVisualization)
            input_return = input("[Press enter to play, type \"q\" to exit]\n")


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
            ax.plot(times[:], us_opt[:, i])
            # hl = ax.hlines(
            #     (self.mpc.stage_factory.u_min[i], self.mpc.stage_factory.u_max[i]), *times[[0, -1]], linestyles="--", colors="r"
            # )
            # handles_.append(hl)
            fontsize = 7
            ax.set_ylabel("$u_{{%d}}$" % (i + 1), fontsize=fontsize)
            ax.tick_params(axis="both", labelsize=fontsize)
            ax.tick_params(axis="y", rotation=90)
            if i + 1 == self.mpc.nu - 1:
                ax.set_xlabel("time", loc="left", fontsize=fontsize)

        ax = plt.subplot(gs[0, 1], projection="3d")
        ax.plot(*pts.T, "r", lw=1.0, label="Actual traj") # actual traj
        ax.plot(*self.mpc.stage_factory.getFullTrajectory(), "g", lw=1, alpha=0.7, label="Command traj") # command trajectory

        for waypoint in waypoints:
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
        plt.plot(times, mpc_loop_times, marker=".", color="dodgerblue", label="Calculation time ")
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
