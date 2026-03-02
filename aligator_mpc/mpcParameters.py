from pathlib import Path
import yaml
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional, Dict, Any, List
from rich import print as richprint
from abc import ABC, abstractmethod
import copy
from typing import Union, Literal, Annotated

# ============================================================================
# Base classes for extensibility
# ============================================================================

class RobotConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(default="UnknownRobot")
    world_frame_name: str = Field(default="Unknown")
    tool_frame_name: str = Field(default="Unknown")
    joints_to_fix: List[str] = Field(default_factory=list)
    
    n_dof: Optional[int] = None
    joint_limits: Optional[Dict[str, tuple]] = None
    speed_limits: Optional[Dict[str, float]] = None
    effort_limits: Optional[Dict[str, float]] = None

    # Metadata for tracking config source
    config_file: Optional[Path] = None

    @model_validator(mode='after')
    def validate_robot_config(self):
        if self.n_dof is None:
            if self.joint_limits:
                self.n_dof = len(self.joint_limits)
            else:
                raise ValueError(f"Robot '{self.name}' must have n_dof or joint_limits")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RobotConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Handle case where robot config is nested under "robot" key in YAML
        if "robot" in data:
            data = data["robot"]
        robot_config = cls(**data)
        robot_config.config_file = Path(path)
        return robot_config

# ============================================================================
# Weights structure
# ============================================================================

# Base composantes for Costs
class RegularisationWeights(BaseModel):
    joint: float = Field(default=None, description="Weight for joint regularisation, default to 0 for no regularisation on joints")
    vel: float = Field(default=None, description="Weight for velocity regularisation, default to 0 for no regularisation on velocities")
    command: float = Field(default=None, description="Weight for command regularisation, default to 0 for no regularisation on commands")

class WaypointWeights6D(BaseModel):
    translation: float = Field(default=None, description="Weight for translation part of the cost, default to 0 for purely orientation tracking")
    orientation: float = Field(default=None, description="Weight for orientation part of the cost, default to 0 for purely position tracking")

class WaypointWeights(BaseModel):
    pose: WaypointWeights6D = Field(default_factory=WaypointWeights6D)
    vel: WaypointWeights6D = Field(default_factory=WaypointWeights6D)

# Base classes for running and terminal costs
class Cost(BaseModel):
    type: str # Discriminator for cost type
    enabled: bool = Field(default=True, description="Whether this cost is active in the problem")

### Running costs
# You can add more cost types as needed (e.g., collision, energy, etc.)
class TrajectoryCost(Cost): # This is the main cost type for trajectory tracking
    type: Literal["trajectory"] = "trajectory"
    regularisation: RegularisationWeights
    waypoints: WaypointWeights

class CollisionCost(Cost): # Exemple d'extension facile
    type: Literal["collision"] = "collision"
    margin: float = Field(default=None, description="Distance margin for collision cost")
    weight: float = Field(default=None, description="Weight for collision cost in the problem")
### Terminal costs
# You can add more cost types as needed (e.g., final pose, final velocity, etc.)
class TerminalCost(Cost):
    type: Literal["terminal"] = "terminal"
    regularisation: RegularisationWeights

# Main Weights class that can be easily extended with new cost types
TermCost = Annotated[
    Union[TerminalCost], 
    Field(discriminator="type")
]
RunCost = Annotated[
    Union[TrajectoryCost, CollisionCost], 
    Field(discriminator="type")
]
class Weights(BaseModel):
    running: Dict[str, RunCost] = Field(default_factory=dict)
    terminal: Dict[str, TermCost] = Field(default_factory=dict)


# ============================================================================
# Solver configuration
# ============================================================================

class SolverParams(BaseModel):
    tolerance: float = Field(default=0.0, description="Tolerance for solver convergence, default to 0 for no tolerance-based stopping criterion")
    mu_init: float = Field(default=None, description="Initial barrier parameter for interior point method, default to 0 for no barrier method")
    max_iters: int = Field(default=100, description="Maximum number of iterations for the solver, default to 100")


class Solver(BaseModel):
    running: SolverParams = Field(default_factory=SolverParams)
    presolve: SolverParams = Field(default_factory=SolverParams)
    rollout_type: str = Field(default="RolloutType.ROLLOUT_RICCATI", description="Type of rollout for MPC iterations")
    sa_strategy: str = Field(default="StrategyType.STRATEGY_LINE_SEARCH", description="Strategy for step acceptance in MPC iterations")
    linear_solver_choice: str = Field(default="LQ_SOLVER_SERIAL", description="Choice of linear solver for Riccati iterations")
    num_threads: int = Field(default=1, description="Number of threads for solver")
    verbose: str = Field(default="aligator.QUIET", description="Verbosity level for solver output")
    
    @model_validator(mode='after')
    def validate_solver_params(self):
        """Ensure presolve has stricter tolerances than running"""
        if self.presolve.tolerance > self.running.tolerance:
            raise ValueError(
                f"Presolve tolerance ({self.presolve.tolerance}) should be stricter than "
                f"running tolerance ({self.running.tolerance}). Adjusting presolve tolerance."
            )
        return self


# ============================================================================
# Reference configuration - trajectory tracking
# ============================================================================

class RegularisationRef(BaseModel):
    """Reference configuration for regularisation costs"""
    joint_pos: List[float] = Field(default_factory=list)
    joint_vel: List[float] = Field(default_factory=list)
    
    @model_validator(mode='after')
    def validate_dimensions(self):
        """Ensure position and velocity vectors have same dimension"""
        if len(self.joint_pos) != len(self.joint_vel):
            raise ValueError(
                f"joint_pos ({len(self.joint_pos)}) and joint_vel "
                f"({len(self.joint_vel)}) must have same dimension"
            )
        return self


# ============================================================================
# Trajectory configuration
# ============================================================================

class Trajectory(BaseModel):
    """Trajectory generation parameters"""
    vel: float = Field(default=0.1, description="Desired velocity for trajectory generation")
    acceleration: Optional[float] = None
    #interpolation_type: str = "cubic"  # cubic, linear, quintic


# ============================================================================
# MPC configuration - main changes
# ============================================================================

class MPC(BaseModel):
    """MPC problem configuration"""
    dt: float = Field(default=0.1, description="Time step for MPC")
    total_time: float = Field(default=5.0, description="Total time horizon for MPC")
    nb_steps_horizon: int = Field(default=20, description="Number of steps in the MPC horizon")
    
    solver: Solver = Field(default_factory=Solver)
    weights: Weights = Field(default_factory=Weights)
    regularisation_ref: RegularisationRef = Field(default_factory=RegularisationRef)
    
    @property
    def n_total_steps(self) -> int:
        """Total number of time steps in the trajectory"""
        if self.total_time is None or self.dt is None:
            raise ValueError("Value of total_time or dt parameter is incorrect")
        return int(self.total_time / self.dt)
    
    @model_validator(mode='after')
    def validate_horizon(self):
        """Validate horizon parameters are consistent"""
        if self.nb_steps_horizon > self.n_total_steps:
            raise ValueError(
                f"nb_steps_horizon ({self.nb_steps_horizon}) cannot be greater than "
                f"n_total_steps ({self.n_total_steps})"
            )
        return self


# ============================================================================
# Main configuration class
# ============================================================================
    

class TaskConfig(BaseModel):
    """Abstract task configuration - easily extensible"""
    name: str = Field(default="UnknownTask", description="Type or name of the task")
    mpc: MPC = Field(default_factory=MPC)
    trajectory: Trajectory = Field(default_factory=Trajectory)
    # Metadata for tracking config source
    config_file: Optional[Path] = None
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "task" in data:
            data = data["task"]
            
        config = cls(**data)
        config.config_file = path
        return config
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary"""
        return cls(**data)

class Config(BaseModel):
    """Main configuration object for MPC"""
    robot: RobotConfig = Field(default_factory=RobotConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    
    # Metadata for tracking config source
    config_name: Optional[str] = None
    
    model_config = ConfigDict(extra="allow")  # Allow additional fields for specific configurations without modifying base class
    
    def merge_with(self, other: "Config") -> "Config":
        """
        Merge two configurations, with other's values overriding self's.
        Useful for overrides per robot or task.
        """
        merged_dict = self.model_dump()
        other_dict = other.model_dump(exclude_none=True)
        
        def deep_merge(base: dict, override: dict) -> dict:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        merged_dict = deep_merge(merged_dict, other_dict)
        return Config(**merged_dict)
    
    def get_robot_config(self) -> RobotConfig:
        """Get robot configuration"""
        return self.robot
    def get_mpc_config(self) -> MPC:
        """Get MPC configuration"""
        return self.task.mpc
    def get_trajectory_config(self) -> Trajectory:
        """Get trajectory configuration"""
        return self.task.trajectory

    def update_weights(self, weights_dict: Dict[str, Any]) -> None:
        """
        Update weights with nested dict support.
        Example: update_weights({
            "running": {"regularisation": {"joint": 2.0}},
            "terminal": {"regularisation": {"command": 0.5}}
        })
        """
        def update_nested(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target.__dict__:
                    update_nested(getattr(target, key), value)
                elif hasattr(target, key):
                    setattr(target, key, value)
        
        update_nested(self.task.mpc.weights, weights_dict)
    
    def create_variant(self, **overrides) -> "Config":
            """
            Create a variant configuration with specific overrides.
            Useful for different robots or tasks.
            """
            return self.model_copy(update=overrides, deep=True)
    
    def __repr__(self) -> str:
        info = f"Config(name={self.config_name}, robot={self.robot.name})"
        if self.config_file:
            info += f" [from {self.config_file}]"
        return info

# ============================================================================
# Configuration manager for project-level settings
# ============================================================================

class ConfigManager:
    """Manage multiple configurations and presets"""
    
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Config] = {}

    def get_robot(self, name: str, path: Optional[Path] = None) -> RobotConfig:
        if path is None:
            robot_path = Path(self.config_dir) / "robots" / f"{name}.yaml"
        else:
            robot_path = Path(path)  / "robots" / f"{name}.yaml"
        return RobotConfig.from_yaml(robot_path)

    def get_task(self, name: str, path: Optional[Path] = None) -> TaskConfig:
        if path is None:
            task_path = Path(self.config_dir) / "tasks" / f"{name}.yaml"
        else:
            task_path = Path(path) / "tasks" / f"{name}.yaml"
        return TaskConfig.from_yaml(task_path)

    def build_full_config(self, robot_name: str, task_name: str, path: Optional[Path] = None) -> Config:
        robot = self.get_robot(robot_name, path)
        task = self.get_task(task_name, path)
        self.configs[f"{robot_name}_{task_name}"] = Config(robot=robot, task=task, config_name=f"{robot_name}_{task_name}")
        return self.configs[f"{robot_name}_{task_name}"]

    def get_config(self, name: str) -> Optional[Config]:
        """Retrieve cached configuration"""
        return self.configs.get(name)


    def list_configs(self) -> List[str]:
        """List all loaded configurations"""
        return list(self.configs.keys())


if __name__ == "__main__":
    # Example usage
    print("\n1. Loading from YAML")
    try:
        path = Path("config")
        robot_config = RobotConfig.from_yaml(path / "robots" / "fr3.yaml")
        print("Robot configuration loaded successfully:")
        richprint(robot_config)
        task_config = TaskConfig.from_yaml(path / "tasks" / "glue_spreading.yaml")
        print("Task configuration loaded successfully:")
        richprint(task_config)
        example_config = Config(robot=robot_config, task=task_config, config_name="fr3_glue_spreading", config_file=[robot_config.config_file, task_config.config_file])
        print("Configuration:")
        richprint(example_config)
    except Exception as e:
        print(f"Error loading configuration: {e}")

    print("\n2. Creating example configuration with config manager")
    config_manager = ConfigManager()
    config = config_manager.build_full_config(robot_name="fr3", task_name="glue_spreading")
    richprint(config)

    print("\n3. Creating a variant configuration for a different trajectory")
    variant = example_config.create_variant(trajectory=Trajectory(vel=0.05))
    richprint(variant)