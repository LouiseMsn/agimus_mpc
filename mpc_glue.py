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
    def __init__(self):
        mpc_parameters = MPCParams()
        print(mpc_parameters)

        self.__stages_model
        pass
    def addJointsLimits(self):
        pass
    def addTorqueLimits(self):
        pass
    def addAutoCollisionsConstraints(self):
        pass
    def getStageModel(self):
        pass

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

        # solver

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
