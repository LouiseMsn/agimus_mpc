import aligator
from aligator import constraints

import pinocchio as pin

from aligator_mpc.mpcParameters import(
    Config,
    StateRegularizationWeights,
    TorqueLimitsConstraint,
    CollisionConstraint,
    ConstraintType,
    CostsConfig,
    WaypointWeightsConfig,
    StateRegularizationWeights
)
from aligator_mpc.mpcUtils import getIndexesFromJointNames
from aligator_mpc.mpcUtils import StagesDefinition
from aligator_mpc.mpcTrajectoryUtils import Interpolator
from aligator_mpc.Costs import Costs
from aligator_mpc.Constraints import Constraints

import numpy as np
from typing import List
from rclpy.impl import rcutils_logger


class StagesFactory():
    def __init__(self, robot, space, n_steps, discrete_dynamics, waypoints, params: Config) -> None:
        self.robot = robot
        self.space = space
        self.nu = self.robot.model.nv
        self.discrete_dynamics = discrete_dynamics
        self.n_steps = n_steps
        self.parameters = params
        self.tool_id = self.robot.model.getFrameId(self.parameters.robot.tool_frame_name)

        # self.stages_definition = StagesDefinition()
        self.stages : list[aligator.stageModel] = []
        self.stage_model_list = []
        self.problem : aligator.TrajOptProblem

        self.waypoints = waypoints

        self.interpolator = self.getInterpolator()

        # self.addConstraints(self.parameters.task.mpc.constraints)
        # self.addCosts(self.parameters.task.mpc.costs)
        self.costs = Costs(self.parameters, robot, space, self.interpolator)
        self.constraints = Constraints(self.parameters, robot, space)
        self.buildStageModelList()

    def getInterpolator(self) -> Interpolator:
        """Builds and returns the interpolator object

        Returns:
            Interpolator: waypoint interpolator
        """
        start_pos = self.robot.data.oMf[self.tool_id]
        self.waypoints = [start_pos] + self.waypoints
        rcutils_logger.RcutilsLogger(name="   MPC_DEBUG   ").info(f'start translation :{ start_pos.translation} rotation: {pin.rpy.matrixToRpy(start_pos.rotation)}')
        return Interpolator(self.waypoints, self.parameters.task.trajectory.vel)

    def getStageModel(self, stage_number:int) -> aligator.StageModel :
        """Returns the stage model for the `stage_number` th stage

        Args:
            stage_number (int): index of the stage to get
        Returns:
            aligator.StageModel : stage model
        """

        if len(self.stage_model_list ) == 0:
            raise ValueError("The stage model list is not built, run buildStageModelList() before running getStageModel()")
        else:
            if stage_number >= self.parameters.task.mpc.n_total_steps :
                return self.stage_model_list[-1]
            else:
                return self.stage_model_list[stage_number]

    def buildStageModelList(self) -> aligator.StageModel:
        """Builds and returns the StageModel for the `stage_number` th stage of the problem.
        """

        for stage_number in range(self.parameters.task.mpc.n_total_steps):
            stage_coststack = aligator.CostStack(self.space, self.nu)
            cost_list = self.getDynamicCosts(stage_number, self.costs.stage_dep_costs)
            for cost in cost_list:
                stage_coststack.addCost(*cost)

            stage_model = aligator.StageModel(stage_coststack, self.discrete_dynamics)
            for constraint in self.constraints.constraints:
                stage_model.addConstraint(*constraint)
            self.stage_model_list.append(stage_model)


    def getTerminalCoststack(self, stage_number:int) -> aligator.CostStack:
        """Returns the terminal coststack calculated from the `stages_definitions` dict.

        Args:
            stage_number (int): index of the stage

        Returns:
            aligator.CostStack: terminal coststack
        """

        terminal_coststack = aligator.CostStack(self.space, self.nu)
        cost_list = self.getDynamicCosts(stage_number, self.costs.terminal_costs)
        for cost in cost_list:
            terminal_coststack.addCost(*cost)

        return terminal_coststack

    def fabricateStages(self, current_stage:int, duration:int) -> tuple[list, aligator.CostStack]:
        """Builds the stages from `current_stage` to `duration` using the costs stored in `self.stages_definition`

        Args:
            current_stage (int): index of the current stage
            duration (_int): duration (in number of indexes)

        Returns:
            List[aligator.StageModel] : list of stages model from index to (index + duration - 1)
            aligator.CostStack : terminal coststack
        """

        terminal_coststack = aligator.CostStack(self.space, self.nu)
        for terminal_cost in self.costs.terminal_costs:
            terminal_coststack.addCost(*terminal_cost)

        stages = []
        for stage_num in range(current_stage, duration + current_stage):
            stage_coststack = aligator.CostStack(self.space, self.nu)
            cost_list = self.getDynamicCosts(stage_num, self.costs.stage_dep_costs)
            if len(self.costs.stage_indep_costs) != 0:
                cost_list = cost_list + self.costs.stage_indep_costs
            for cost in cost_list:
                stage_coststack.addCost(*cost)

            stage_model = aligator.StageModel(stage_coststack, self.discrete_dynamics)
            for constraint in self.constraints.constraints:
                stage_model.addConstraint(*constraint)
            stages.append(stage_model)
        return stages , terminal_coststack
    # ==========================================================================
    # Utils
    # ==========================================================================

    def getDynamicCosts(self, stage_number:int, master_cost_list:list) -> list:
        """Returns a list of costs for the current stage. If a cost is not defined for a given `stage_number`, the last instance of this cost is returned instead.

        Args:
            stage_number (int): index of the stage's cost
            master_cost_list (list): list where all the costs are stored

        Returns:
            list: list of costs to apply during stage_number
        """
        return_costs_list = []
        for cost_list in master_cost_list :
            if stage_number < len(cost_list):
                return_costs_list.append(cost_list[stage_number])
            else:
                return_costs_list.append(cost_list[-1])
        return return_costs_list

    def getFullTrajectory_pt_by_pt(self) -> list:  # TODO : move to pattern generator?
        """Return the full trajectory formatted as np.array([x0,y0,z0], [x1,y1,z1], ...)

        Returns:
            list: list of points
        """
        traj = []
        for i in range(self.parameters.task.mpc.n_total_steps):
            pos, _ = self.interpolator(i*self.parameters.task.mpc.dt)
            traj.append(pos.translation)
        return traj
