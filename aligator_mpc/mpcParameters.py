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

class RegularisationRef(BaseModel):
    joint_pos: list
    joint_vel: list

class MPC(BaseModel):
    dt: float
    total_time: int
    nb_steps_horizon: int
    solver: Solver
    weights: Weights
    regularisation_ref: RegularisationRef

    @property
    def n_total_steps(self):
        if self.total_time is None or self.dt is None:
            raise ValueError("Value of total_time or dt parameter is incorrect")
        return int(self.total_time / self.dt)

class Robot(BaseModel):
    name: str
    world_frame_name: str
    tool_frame_name: str
    joints_to_fix: list

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

if __name__=="__main__":
    path = Path("config/mpc_config.yaml")
    params_test = Config.from_yaml(path)

    richprint(params_test)