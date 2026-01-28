from pathlib import Path
import yaml
from pydantic import BaseModel
from rich import print as richprint


class RegularisationWeights(BaseModel):
    joint: float
    vel: float
    command: float

class WaypointWeights6D(BaseModel):
    translation: float
    orientation: float

class WaypointWeights(BaseModel):
    pose: WaypointWeights6D
    vel: WaypointWeights6D

class RunningWeights(BaseModel):
    regularisation: RegularisationWeights
    waypoints: WaypointWeights

class TerminalWeights(BaseModel):
    regularisation: RegularisationWeights

class Weights(BaseModel):
    running: RunningWeights
    terminal: TerminalWeights
    collision: float

class SolverParams(BaseModel):
    tolerance: float
    mu_init: float
    max_iters : int

class Solver(BaseModel):
    running : SolverParams
    presolve : SolverParams
    rollout_type: str
    sa_strategy: str
    linear_solver_choice: str
    num_threads: int
    verbose : str

class MPC(BaseModel):
    dt: float
    total_time: int
    nb_steps_horizon: int
    solver: Solver
    weights: Weights

    @property
    def n_total_steps(self):
        if self.total_time is None or self.dt is None:
            raise ValueError("Value of total_time or dt parameter is incorrect")
        return int(self.total_time / self.dt)


class Robot(BaseModel):
    name: str
    world_frame_name: str
    tool_frame_name: str

class Trajectory(BaseModel):
    vel: float


class Config(BaseModel):
    robot: Robot
    mpc: MPC
    trajectory: Trajectory

    @classmethod
    def from_yaml(cls, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)




    # def __repr__(self)->str:
    #     """
    #     Formats the output when printing the object
    #     """
    #     return f'Robot parameters:\n'\
    #                 f'\tRobot name: {self.robot_name}\n'\
    #                 f'\tWorld frame name: "{self.world_frame_name}"\n'\
    #                 f'\tTool frame name: "{self.tool_frame_name}"\n'\
    #             f'\nMPC parameters:\n'\
    #                 f'\tdt: {self.dt} (secs)\n'\
    #                 f'\tTotal time: {self.total_time} (secs)\n'\
    #                 f'\tTotal number of steps: {self.n_total_steps}\n'\
    #                 f'\tHorizon: {self.nb_steps_horizon} (steps)\n'\
    #                 f'\tHorizon: {self.mpc_horizon} (secs)\n'\
    #                 f'\tNumber max of iterations at 1st iteration: {self.solver.max_iters_1st_iter}\n'\
    #                 f'\tNumber max of iterations: {self.solver.max_iters}\n'\
    #                 f'\tSolver:\n'\
    #                     f'\t\tVerbose: {self.verbose}\n'\
    #                     f'\t\tTolerance: {self.solver.tolerance}\n'\
    #                     f'\t\tRollout type: {self.solver.rollout_type}\n'\
    #                     f'\t\tSA Strategy: {self.solver.sa_strategy}\n'\
    #                     f'\t\tLinear solver choice: {self.solver.linear_solver_choice}\n'\
    #                     f'\t\tNumber of threads: {self.solver.num_threads}\n'\
    #                     f'\t\tMu initialization at 1rst iteration: {self.solver.mu_init_1st_iter}\n'\
    #                     f'\t\tMu at initialization: {self.solver.mu_init}\n'\
    #             f'\nWeights parameters:\n'\
    #             f'\tRegularisations costs:\n'\
    #                 f'\t\tJoints: {self.mpc.weights.running.regularisation.joint}\n'\
    #                 f'\t\tVelocity: {self.mpc.weights.running.regularisation.vel}\n'\
    #                 f'\t\tCommand: {self.mpc.weights.running.regularisation.command}\n'\
    #                 f'\t\tTerminal state: {self.term_state_reg_cost}\n'\
    #             f'\n\tWaypoints:\n'\
    #                 f'\t\tFrame position: {self.mpc.weights.waypoints.frame_pos}\n'\
    #                 f'\t\tFrame velocity: {self.mpc.weights.waypoints.frame_vel}\n'\
    #             f'\n\tOrientation: {self.mpc.weights.waypoints.orientation}\n'\
    #             f'\n\tCollision: {self.mpc.weights.collision}\n'


if __name__=="__main__":
    path = Path("config/mpc_config.yaml")
    params_test = Config.from_yaml(path)
    # print(params_test.robot)

    richprint(params_test)

