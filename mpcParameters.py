import numpy as np
import aligator
import tap

class ArgsBase(tap.Tap):
    display: bool = False  # Displays the trajectory using meshcat

class Args(ArgsBase):
    debug : bool = False # Adds prints
    viz3D : bool = False # Displays a 3D visualization
    perturbate : bool = False # Adds a perturbation to the state input of the MPC
    no_joints_lim: bool = False
    no_torque_lim: bool = False
    no_orientation_cost : bool = False
    no_waypoints : bool = False

args = Args().parse_args()

class Params():
    """
    Class regrouping the parameters used in the MPC
    """
    def __init__(self)->None:
        # Robot
        self.robot_name : str = "panda"
        self.world_frame_name : str = "universe"
        self.start_pose : np.ndarray = np.array([0, -1, 0, -2.5, 0.0, 2, 0.0, 0.0, 0.0])
        self.tool_frame_name : str = "panda_hand_tcp"

        # MPC
        self.dt : float = 0.01 # time is in seconds
        self.total_time : int|float = 10
        self.mpc_steps : int = 100
        self.mpc_max_iter : int = 1 #2
        self.solver_tolerance = 1e-7
        self.solver_rollout_type = aligator.ROLLOUT_LINEAR
        self.solver_sa_strategy = aligator.SA_LINESEARCH_NONMONOTONE
        self.solver_linear_solver_choice = aligator.LQ_SOLVER_PARALLEL
        self.solver_num_threads = 10
        self.mu_init = 1e-7 #0.99 # penalite sur les contraintes
        if args.debug:
            self.verbose = aligator.VerboseLevel.VERBOSE
        else:
            self.verbose = aligator.VerboseLevel.QUIET

        # Weights:
        self.stage_joint_reg_cost = 1e-2 #1e-4
        self.stage_vel_reg_cost = 1e-2
        self.command_reg_cost = 1e-2
        self.term_state_reg_cost = 1e-4
        self.waypoint_x_weight = 1e-4
        self.waypoint_frame_pos_weight = 100
        self.waypoint_frame_vel_weight = 1
        self.orientation_weight = 1
        self.vel_spread = 0.5
        self.vel_start = 0.5

        # Trajectory
        self.tool_orientation = np.array([np.pi, 0., 0.])

    @property
    def n_total_steps(self):
        return int(self.total_time / self.dt)

    @property
    def mpc_horizon(self):
        return self.dt * float(self.mpc_steps) # in seconds

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
                    f'\tHorizon: {self.mpc_steps} (steps)\n'\
                    f'\tHorizon: {self.mpc_horizon} (secs)\n'\
                    f'\tNumber max of iterations: {self.mpc_max_iter}\n'\
                    f'\tSolver:\n'\
                        f'\t\tRollout type: {self.solver_rollout_type}\n'\
                        f'\t\tSA Strategy: {self.solver_sa_strategy}\n'\
                        f'\t\tLinear solver choice: {self.solver_linear_solver_choice}\n'\
                        f'\t\tNumber of threads: {self.solver_num_threads}\n'\
                f'\nWeights parameters:\n'\
                f'\tRegulations costs:\n'\
                    f'\t\tJoints: {self.stage_joint_reg_cost}\n'\
                    f'\t\tVelocity: {self.stage_vel_reg_cost}\n'\
                    f'\t\tCommand: {self.command_reg_cost}\n'\
                    f'\t\tTerminal state: {self.term_state_reg_cost}\n'\
                f'\n\tWaypoints:\n'\
                    f'\t\tState: {self.waypoint_x_weight}\n'\
                    f'\t\tFrame position: {self.waypoint_frame_pos_weight}\n'\
                    f'\t\tFrame velocity: {self.waypoint_frame_vel_weight}\n'\
                f'\n\tOrientation: {self.orientation_weight}\n'
