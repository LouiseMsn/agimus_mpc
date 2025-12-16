import numpy as np
import aligator
from pathlib import Path
import yaml


class Args():
    debug : bool = False # Adds prints
    no_3Dviz : bool = False # Displays a 3D visualization
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
    def __init__(self, yaml_config_path)->None:
        with yaml_config_path.open('r') as config_file:
            config = yaml.safe_load(config_file)

            # Robot
            robot = config["robot"]
            self.robot_name = robot['name']
            self.world_frame_name = robot['world_frame_name']
            self.start_pose = eval(robot['start_pose'])
            self.tool_frame_name = robot['tool_frame_name']

            # MPC
            mpc = config['MPC']
            self.dt = mpc['dt'] # time is in seconds
            self.total_time = mpc['total_time']
            self.nb_steps_horizon = mpc['nb_steps_horizon']
            self.mpc_max_iter = mpc['max_nb_iter']
            solver = mpc['solver']
            self.solver_tolerance = eval(solver['tolerance'])
            self.solver_rollout_type = eval(solver['rollout_type'])
            self.solver_sa_strategy = eval(solver['sa_strategy'])
            self.solver_linear_solver_choice = eval(solver['linear_solver_choice'])
            self.solver_num_threads = solver['num_threads']
            self.mu_init = eval(solver['mu_init'])  # penality on constraints
            if args.debug:
                self.verbose = aligator.VerboseLevel.VERBOSE
            else:
                self.verbose = aligator.VerboseLevel.QUIET

            # Weights:
            weights = mpc['weights']
            reg_weights = weights['regulation']
            self.stage_joint_reg_cost = eval(reg_weights['stage_joint'])
            self.stage_vel_reg_cost = eval(reg_weights['stage_vel'])
            self.command_reg_cost = eval(reg_weights['command'])
            self.term_state_reg_cost = eval(reg_weights['term_state'])

            waypt_weights = weights['waypoints']
            self.waypoint_frame_pos_weight = waypt_weights['frame_pos']
            self.waypoint_frame_vel_weight = waypt_weights['frame_vel']
            self.orientation_weight = waypt_weights['orientation']

            self.collision_weight = weights['collision'] # very sensitive, max around ~ 0.1

            # Trajectory
            trajectory = config['trajectory']
            self.tool_orientation = trajectory['tool_orientation']
            velocity = trajectory['velocity']
            self.vel_spread = velocity['spread']
            self.vel_start = velocity['start']


    @property
    def n_total_steps(self):
        if self.total_time is None or self.dt is None:
            raise ValueError("Value of total_time or dt parameter is incorrect")
        return int(self.total_time / self.dt)

    @property
    def mpc_horizon(self):
        if self.dt is None or self.nb_steps_horizon is None:
            raise ValueError("Value of dt or nb_steps_horizon is incorrect")
        return self.dt * float(self.nb_steps_horizon) # in seconds

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
                    f'\tHorizon: {self.nb_steps_horizon} (steps)\n'\
                    f'\tHorizon: {self.mpc_horizon} (secs)\n'\
                    f'\tNumber max of iterations: {self.mpc_max_iter}\n'\
                    f'\tSolver:\n'\
                        f'\t\tTolerance: {self.solver_tolerance}\n'\
                        f'\t\tRollout type: {self.solver_rollout_type}\n'\
                        f'\t\tSA Strategy: {self.solver_sa_strategy}\n'\
                        f'\t\tLinear solver choice: {self.solver_linear_solver_choice}\n'\
                        f'\t\tNumber of threads: {self.solver_num_threads}\n'\
                        f'\t\tMu at initialization: {self.mu_init}\n'\
                f'\nWeights parameters:\n'\
                f'\tRegulations costs:\n'\
                    f'\t\tJoints: {self.stage_joint_reg_cost}\n'\
                    f'\t\tVelocity: {self.stage_vel_reg_cost}\n'\
                    f'\t\tCommand: {self.command_reg_cost}\n'\
                    f'\t\tTerminal state: {self.term_state_reg_cost}\n'\
                f'\n\tWaypoints:\n'\
                    f'\t\tFrame position: {self.waypoint_frame_pos_weight}\n'\
                    f'\t\tFrame velocity: {self.waypoint_frame_vel_weight}\n'\
                f'\n\tOrientation: {self.orientation_weight}\n'\
                f'\n\tCollision: {self.collision_weight}\n'


if __name__=="__main__":
    path = Path("config/mpc_config.yaml")
    params_test = Params(path)
    print(params_test)
