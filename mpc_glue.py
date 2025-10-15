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
        pass


class BaseStageFactory():
    def __init__(self):
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
