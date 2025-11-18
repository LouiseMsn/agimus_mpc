from mpc import MPC
from mpcParameters import Params
from mpcTrajectoryUtils import PatternGenerator
import aligator
import matplotlib.pyplot as plt
import numpy as np
import os
from statistics import median

# os.nice(-20)

def runBenchParallel(nb_threads, nb_stages):
    print(f"nb threads {nb_threads}, nb stages {nb_stages}")
    params= Params()
    params.solver_linear_solver_choice = aligator.LQ_SOLVER_PARALLEL
    params.solver_num_threads = nb_threads
    # params.mpc_steps = nb_stages
    mpc = MPC(parameters=params, waypoints=positions)
    # delta_t = mpc.calcNextCommand(t=0, current_xs=None)
    delta_t = mpc.bench(nb_stages)

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
    num_threads_list = list(range(2,13,2))
    stages_axis = [i for i in range(50, 201,50)]
    moy = 3

    fig, ax = plt.subplots()
    list_results = []
    for num_threads in num_threads_list:
        list_delta_t = []
        for num_stages in stages_axis:
            d_ts = []
            for i in range(moy):
                d_ts.append(runBenchParallel(num_threads, num_stages))
                # d_t = d_t + runBenchSerial(num_stages)
            list_delta_t.append(median(d_ts))
        ax.plot(stages_axis, list_delta_t, marker=".", label=f"{num_threads} threads")
        list_results.append(list_delta_t)

    ax.legend()
    ax.set(title='Aligator 0.16.0', xlabel='num of stages', ylabel = 'time (s)' )

    pivoted_results = list(zip(*list_results))

    # Now plot delta as a function of number of threads for a given stage
    fig, ax = plt.subplots()

    for stage_idx, stage_times in enumerate(pivoted_results):
        ax.plot(num_threads_list, stage_times, marker="o", label=f"stage {stages_axis[stage_idx]}")

    ax.set(title='Aligator 0.16.0', xlabel='num of threads', ylabel='time (s)')
    ax.legend()
    plt.show()
