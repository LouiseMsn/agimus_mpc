import aligator
from aligator import constraints, manifolds, dynamics

import example_robot_data as ex_robot_data
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import hppfcl

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Literal, List, Optional
import tap

import time
import sys

class MPC():
    def __init__(self, parameters):
        self.robot = ex_robot_data.load()
        stage_factory = GlueStageFactory()


class BaseStageFactory():
    def __init__(self, robot_model, space):
        self.robot_model = robot_model
        self.space = space
        self.nv = self.robot_model.nv
        self.nu = self.nv
        self.ndx = self.space.ndx

        self.parameters = ProblemParams()
        print(self.parameters)

        self.mpc_parameters = MPCParams()

        self.__stages_definition: dict[str,list[tuple]] = {"Contraints":[],"Global costs":[], "Stage dependant costs":[]} # Constraints are all global
        self.__stages : list[aligator.stageModel] = []
        self.__problem : aligator.TrajOptProblem = None

    def fabricate_stages(self, current_t, n_steps,)->list[aligator.stageModel]:
        """
        TODO
        """
        return 0

    def addJointsLimitsConstraints(self) -> None:
        """
        Adds joints limits contraints (joint_angle_residual, box_constraint) to self.__stages_definition["Contraints"]
        """


        # Résidu de base qui extrait simplement l'état x.
        # C'est f(x) = x - 0, donc il renvoie x.
        identity_residual = aligator.StateErrorResidual(self.space, self.nu, self.space.neutral())

        for j_idx, jn in enumerate(self.robot_model.names[1:]):
            q_idx_in_x = self.robot_model.joints[j_idx + 1].idx_q
            q_min = self.robot_model.lowerPositionLimit[q_idx_in_x]
            q_max = self.robot_model.upperPositionLimit[q_idx_in_x]

            if np.isneginf(q_min) and np.isinf(q_max):
                continue

            print(f"Adding BoxConstraint for Joint '{jn}': [{q_min:.3f}, {q_max:.3f}]")

            A = np.zeros((1, self.ndx))
            A[0, q_idx_in_x] = 1.0
            b = np.array([0.0])
            joint_angle_residual = aligator.LinearFunctionComposition(identity_residual, A, b)

            box_constraint = constraints.BoxConstraint(np.array([q_min]), np.array([q_max]))

            constraint = (joint_angle_residual, box_constraint)
            self.__stages_definition["Contraints"].append(constraint)

    def addTorqueLimitsConstraints(self):
        """
        Adds torque limits contraints (joint_angle_residual, box_constraint) to self.__stages_definition["Contraints"]
        """
        nv = self.robot_model.nv
        nu = nv
        ndx = self.space.ndx
        self.u_max = self.robot_model.effortLimit
        self.u_min = -self.u_max
        residual = aligator.ControlErrorResidual(ndx, nu)
        constraint = constraints.BoxConstraint(self.u_min, self.u_max)
        self.__stages_definition["Constraints"].append((residual, constraint))

    def addRegulationCosts(self):
        wt_x = self.parameters.joint_reg_cost*np.ones(self.ndx)
        wt_x[self.nv:] = self.parameters.vel_reg_cost
        wt_x = np.diag(wt_x)
        wt_u = self.parameters.command_reg_cost*np.eye(self.nu)

        # add Global target
        wt_x_term = wt_x.copy() # np.ones(self.n_dx)
        wt_x_term[:] = self.parameters.term_state_reg_cost

        global_cost = ("reg", aligator.QuadraticCost(wt_x_term, wt_u * 0))
        self.__stages_definition["Global costs"].append(global_cost)

        cost_per_stage = ("reg", aligator.QuadraticCost(wt_x * self.mpc_parameters.dt, wt_u * self.mpc_parameters.dt)) #! comment gérer la diff entre une reg a chaque point qui est statique et un cost de target qui bouge avec t
        self.__stages_definition["Stage dependant costs"].append(cost_per_stage)


    def addAutoCollisionsConstraints(self):
        pass

    def getStageModel(self):
        return self.__stages

class GlueStageFactory(BaseStageFactory):
    def __init__(self):
        super().__init__()

# TODO class visu?

class MPCParams():
    def __init__(self)->None:
        # Robot
        self.robot_name : str = "panda"
        self.world_frame_name : str = "universe"
        self.start_pose : np.ndarray = np.array([0, 0.0, 0.0, -1, 0.0, 2, 0.0, 0.0, 0.0])
        self.tool_frame_name : str = "panda_hand_tcp"

        # MPC
        self.dt : float = 0.01 # time is in seconds
        self.total_time : int = 1
        self.n_total_steps : int = int(self.total_time / self.dt)

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
                    f'\tNumber of steps: {self.n_total_steps}\n'\


class ProblemParams():
    def __init__(self)->None:
        # Weights:
        self.joint_reg_cost = 1e-2 #1e-4
        self.vel_reg_cost = 1e-2
        self.command_reg_cost = 1e-4
        self.term_state_reg_cost = 1e-4
        self.waypoint_x_weight = 1e-4
        self.waypoint_frame_pos_weight = 200.0
        self.waypoint_frame_vel_weight = 1
        self.orientation_weight = 50

        # Constraints


    def __repr__(self):
        """
        Formats the output when printing the object
        """
        return f'Weights parameters:\n'\
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
    params = MPCParams()
    print(params)
