from mpc import MPC, Visualization
from mpcParameters import Params
from mpcTrajectoryUtils import PatternGenerator
import numpy as np
from copy import deepcopy

if __name__=="__main__":


    parameters = Params()
    patternGen = PatternGenerator([0.5,0.5,0], (0.4,0,0))
    x,y,z = patternGen.generate_pattern('zigzag_curve',stride=0.05)
    positions :list = []
    for i in range (len(x)): # TODO change return of generate pattern to avoid this
        positions.append(np.array([x[i], y[i], z[i]]))

    results_xs = []
    results_us = []
    prim_infeas = []
    dual_infeas = []
    mpc_timer = []


    mpc = MPC(positions, parameters)
    viz = Visualization(mpc)
    launch_check = input("Enter to launch")
    robot_state = mpc.x0
    for t in range (mpc.parameters.n_total_steps+1):
        print(f't:{t}')
        if t == 0:
            robot_state = mpc.x0
        else:
            robot_state = mpc.results.xs.tolist()[0]
        solver_calc_time = mpc.iterate(robot_state)

        # removing the actuation on the two grips of the end effector
        xs_no_ee = mpc.results.xs.tolist()
        xs_no_ee[0][mpc.n_q - 1] = 0
        xs_no_ee[0][mpc.n_q - 2] = 0

        viz.display_step(xs_no_ee)

        # copy the results for plotting
        current_xs = deepcopy(mpc.results.xs.tolist()[0])
        current_us = deepcopy(mpc.results.us.tolist()[0])
        last_dual_infeas = deepcopy(mpc.callback.dual_infeas.tolist()[-1])
        last_prim_infeas = deepcopy(mpc.callback.prim_infeas.tolist()[-1])

        results_us.append(current_us)
        results_xs.append(current_xs)
        prim_infeas.append(last_dual_infeas)
        dual_infeas.append(last_prim_infeas)
        mpc_timer.append(solver_calc_time)

    viz.plotResults(results_xs, results_us, prim_infeas, dual_infeas, mpc_timer)
