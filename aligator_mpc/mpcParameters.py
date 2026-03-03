from pathlib import Path
import yaml
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional, Dict, Any, List
from rich import print as richprint
from abc import ABC, abstractmethod
import copy
from typing import Union, Literal, Annotated, Tuple

# ============================================================================
# Base classes
# ============================================================================

class RobotConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    world_frame_name: str
    tool_frame_name: str
    locked_joints: List[str] = Field(default_factory=list)
    meshes_packages: List[str] = Field(default_factory=list)
    
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
        robot_config = cls(**data)
        if robot_config.name == None:
            raise ValueError(f"Robot configuration loaded from {path} must have a name")
        if robot_config.world_frame_name == None:
            raise ValueError(f"Robot configuration loaded from {path} must have a world_frame_name")
        if robot_config.tool_frame_name == None:
            raise ValueError(f"Robot configuration loaded from {path} must have a tool_frame_name")
        robot_config.config_file = Path(path)
        return robot_config
    
# ============================================================================
# Constraints Configuration
# ============================================================================

class Constraint(BaseModel):
    """Base constraint class"""
    type: str
    enabled: bool = Field(default=True)

class JointLimitsConstraint(Constraint):
    type: Literal["joint_limits"] = "joint_limits"
    #? ajouter des marges plus petites que les marges constructeur ??

class TorqueLimitsConstraint(Constraint):
    type: Literal["torque_limits"] = "torque_limits"
    scale_factor: float = Field(
        default=1.0,
        description="Scale effort limits (e.g., 0.8 for 80% safety margin)"
    )
    per_joint_scaling: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-joint effort scaling factors"
    )

class VelocityConstraint(Constraint):
    type: Literal["velocity"] = "velocity"
    scale_factor: float = Field(
        default=1.0,
        description="Scale effort limits (e.g., 0.8 for 80% safety margin)"
    )
    per_joint_scaling: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-joint effort scaling factors"
    )

class CollisionConstraint(Constraint):
    type: Literal["collision"] = "collision"
    margin: float = Field(
        default=0.02,
        description="Minimum allowed distance (m)"
    )
    pairs: Optional[Tuple[str, str]] = Field(
        default=None,
        description="Specific collision pairs to check"
    )
# Pydantics does not handle using `Contraint` type for the inherited classes
ConstraintType = Annotated[
    Union[JointLimitsConstraint, TorqueLimitsConstraint, VelocityConstraint, CollisionConstraint],
    Field(discriminator="type")
]

class ConstraintsConfig(BaseModel):
    """All constraints for the MPC problem"""
    constraints: List[ConstraintType] = Field(
        default_factory=list,
        description="List of constraints"
    )
    
    def get_enabled_constraints(self) -> List[ConstraintType]:
        """Return only enabled constraints"""
        return [c for c in self.constraints if c.enabled]
    
# ============================================================================
# Weights structure
# ============================================================================

class WeightVector(BaseModel):
    """Flexible weight definition"""
    mode: Literal["vector", "scale", "identity"] = Field(
        default="vector",
        description="'vector'=explicit values, 'scale'=uniform scaling, 'identity'=1.0"
    )
    
    values: Optional[List[float]] = Field(
        default=None,
        description="Explicit weight values (for mode='vector')"
    )
    
    scale: Optional[float] = Field(
        default=1.0,
        description="Uniform scaling factor"
    )

    @model_validator(mode='after')
    def validate_weights(self):
        if self.mode == "vector" and self.values is None:
            raise ValueError("'vector' mode requires 'values' field")
        return self

class StateRegularizationWeights(BaseModel):
    """Weights for state regularization"""
    position: WeightVector = Field(
        default_factory=WeightVector,
        description="Position regularization weights"
    )
    velocity: WeightVector = Field(
        default_factory=WeightVector,
        description="Velocity regularization weights"
    )
    torque: WeightVector = Field(
        default_factory=WeightVector,
        description="Torque regularization weights"
    )

class WaypointWeights6D(BaseModel):
    """6D waypoint weights (translation + rotation)"""
    translation: List[float] = Field(default=[1.0, 1.0, 1.0], description="Weights for translation part of the cost")
    orientation: List[float] = Field(default=[1.0, 1.0, 1.0], description="Weights for orientation part of the cost")

class WaypointWeightsConfig(BaseModel):
    """Waypoint tracking weights"""
    pose: WaypointWeights6D = Field(default_factory=WaypointWeights6D)
    velocity: WaypointWeights6D = Field(default_factory=WaypointWeights6D)

class Cost(BaseModel):
    """Base cost class"""
    type: str
    enabled: bool = Field(default=True)

class TrajectoryCost(Cost):
    """Main trajectory tracking cost"""
    type: Literal["trajectory"] = "trajectory"
    
    state_regularization: StateRegularizationWeights = Field(
        default_factory=StateRegularizationWeights
    )
    waypoints: WaypointWeightsConfig = Field(
        default_factory=WaypointWeightsConfig
    )

class TerminalCost(Cost):
    """Terminal state cost"""
    type: Literal["terminal"] = "terminal"
    
    state_regularization: StateRegularizationWeights = Field(
        default_factory=StateRegularizationWeights
    )

RunCostType = Annotated[
    Union[TrajectoryCost],
    Field(discriminator="type")
]

TermCostType = Annotated[
    Union[TerminalCost],
    Field(discriminator="type")
]

class CostsConfig(BaseModel):
    """All costs in the MPC problem"""
    running: Dict[str, RunCostType] = Field(
        default_factory=dict,
        description="Running (stage) costs"
    )
    terminal: Dict[str, TermCostType] = Field(
        default_factory=dict,
        description="Terminal cost"
    )

# ============================================================================
# Solver configuration
# ============================================================================

class SolverParams(BaseModel):
    tolerance: float = Field(description="Tolerance for solver convergence")
    mu_init: float = Field(description="Initial barrier parameter for interior point method")
    max_iters: int = Field(description="Maximum number of iterations for the solver")


class Solver(BaseModel):
    running: SolverParams = Field(default_factory=SolverParams)
    presolve: SolverParams = Field(default_factory=SolverParams)
    rollout_type: str = Field(description="Type of rollout for MPC iterations")
    sa_strategy: str = Field(description="Strategy for step acceptance in MPC iterations")
    linear_solver_choice: str = Field(description="Choice of linear solver for Riccati iterations")
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
        if self.num_threads < 1:
            raise ValueError("num_threads must be >= 1")
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
    vel: float = Field(description="Desired velocity for trajectory generation")
    acceleration: Optional[float] = None
    #interpolation_type: str = "cubic"  # cubic, linear, quintic


# ============================================================================
# MPC configuration - main changes
# ============================================================================

class MPC(BaseModel):
    """MPC problem configuration"""
    dt: float = Field(description="Time step for MPC")
    total_time: float = Field(description="Total time horizon for MPC")
    nb_steps_horizon: int = Field(description="Number of steps in the MPC horizon")
    
    solver: Solver = Field(default_factory=Solver)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    regularisation_ref: RegularisationRef = Field(default_factory=RegularisationRef)
    
    @property
    def n_total_steps(self) -> int:
        """Total number of time steps in the trajectory"""
        if self.total_time is None or self.dt is None:
            raise ValueError("Value of total_time or dt parameter is incorrect")
        return int(self.total_time / self.dt)
    
    @model_validator(mode='after')
    def validate_mpc(self):
        """Validate all MPC parameters"""
        # Time parameters
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        
        if self.total_time <= 0:
            raise ValueError(f"total_time must be positive, got {self.total_time}")
        
        if self.nb_steps_horizon <= 0:
            raise ValueError(f"nb_steps_horizon must be positive")
        
        # Consistency checks
        actual_horizon_time = self.nb_steps_horizon * self.dt
        if actual_horizon_time > self.total_time * 1.01:  # Allow 1% tolerance for rounding
            raise ValueError(
                f"Horizon time ({actual_horizon_time:.3f}s) exceeds total_time ({self.total_time}s)"
            )
        
        if self.nb_steps_horizon > self.n_total_steps:
            raise ValueError(
                f"Horizon steps ({self.nb_steps_horizon}) > total steps ({self.n_total_steps})"
            )
        
        return self


# ============================================================================
# Main configuration class
# ============================================================================
    

class TaskConfig(BaseModel):
    """Abstract task configuration - easily extensible"""
    name: str = Field(description="Type or name of the task")
    mpc: MPC = Field(default_factory=MPC)
    trajectory: Trajectory = Field(default_factory=Trajectory)
    config_file: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config = cls(**data)
        config.config_file = path
        return config


class Config(BaseModel):
    """Main configuration object for MPC"""
    robot: RobotConfig = Field(default_factory=RobotConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    
    # Metadata for tracking config source
    config_name: Optional[str] = None
    
    model_config = ConfigDict(extra="allow")  # Allow additional fields for specific configurations without modifying base class
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load from single YAML file"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
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

    def load_robot(self, name: str, path: Optional[Path] = None) -> RobotConfig:
        """Load robot configuration"""
        if path is None:
            robot_path = self.config_dir / "robots" / f"{name}.yaml"
        else:
            robot_path = Path(path) / "robots" / f"{name}.yaml"
        
        return RobotConfig.from_yaml(robot_path)
    
    def load_task(self, name: str, path: Optional[Path] = None) -> TaskConfig:
        """Load task configuration"""
        if path is None:
            task_path = self.config_dir / "tasks" / f"{name}.yaml"
        else:
            task_path = Path(path) / "tasks" / f"{name}.yaml"
        
        return TaskConfig.from_yaml(task_path)

    def build_config(self, robot_name: str, task_name: str, path: Optional[Path] = None) -> Config:
        """Build complete configuration"""
        robot = self.load_robot(robot_name, path)
        task = self.load_task(task_name, path)
        
        config_key = f"{robot_name}_{task_name}"
        config = Config(robot=robot, task=task, config_name=config_key)
        self.configs[config_key] = config
        
        return config
    def add_config(self, config: Config) -> None:
        """Add a pre-built configuration to the manager"""
        if config.config_name is None:
            raise ValueError("Config must have a config_name to be added to ConfigManager")
        if config.config_name in self.configs:
            raise ValueError(f"Config with name '{config.config_name}' already exists in ConfigManager")
        self.configs[config.config_name] = config
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
    config = config_manager.build_config(robot_name="fr3", task_name="glue_spreading")
    richprint(config)

    print("\n3. Creating a variant configuration for a different trajectory")
    variant = example_config.create_variant(trajectory=Trajectory(vel=0.05))
    richprint(variant)