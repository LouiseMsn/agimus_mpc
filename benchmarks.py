from mpc import MPC
from mpcParameters import Params
from mpcTrajectoryUtils import PatternGenerator
import aligator
import matplotlib.pyplot as plt
import numpy as np


MIN = 50
MAX = 150

def runBenchParallel(nb_threads, nb_stages):
    params= Params()
    params.solver_linear_solver_choice = aligator.LQ_SOLVER_PARALLEL
    params.solver_num_threads = nb_threads
    print(nb_stages)
    params.mpc_steps = nb_stages
    mpc = MPC(parameters=params, waypoints=positions)
    delta_t = mpc.calcNextCommand(t=0, current_xs=None)

    return delta_t

def runBenchSerial(nb_stages):
    params= Params()
    params.solver_linear_solver_choice = aligator.LQ_SOLVER_SERIAL
    print(nb_stages)
    params.mpc_steps = nb_stages
    mpc = MPC(parameters=params, waypoints=positions)
    delta_t = mpc.calcNextCommand(t=0, current_xs=None)

    return delta_t

if __name__=="__main__":

    patternGen = PatternGenerator([0.5,0.5,0], (0.5,0,0.1))
    x,y,z = patternGen.generate_pattern('zigzag_curve',stride=0.1)
    positions :list = []
    for i in range (len(x)):
        positions.append(np.array([x[i], y[i], z[i]]))

    # params = Params()
    stages_axis = [i for i in range(MIN, MAX,10)]

    fig, ax = plt.subplots()
    list_results = []
    for num_threads in range(2,9,2):
        list_delta_t = []
        for num_stages in range(MIN,MAX,10):
            d_t = 0
            for i in range(1):
                d_t = d_t + runBenchParallel(num_threads, num_stages)
                # d_t = d_t + runBenchSerial(num_stages)
            list_delta_t.append(d_t/1.0)
        ax.plot(stages_axis, list_delta_t, marker=".", label=f"{num_threads} threads")
    list_results.append(list_delta_t)

    ax.legend()
    ax.set(title='Aligator 0.16.0', xlabel='num of stages', ylabel = 'time (s)' )
    plt.show()
