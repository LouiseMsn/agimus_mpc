from mpc import MPC
from mpcVisualization import Visualization
from mpcParameters import Params, args
from mpcTrajectoryUtils import PatternGenerator
import numpy as np
from copy import deepcopy
import pinocchio as pin

if __name__=="__main__":
    parameters = Params()

    # Waypoints ================================================================
    patternGen = PatternGenerator([0.5,0.5,0], (0.6,0,0.3))
    positions = patternGen.generate_pattern('zigzag_curve',stride=0.05)

    # positions =[ # line²
    #             np.array([ 0.3, 0, 0.2]),
    #             np.array([ 0.2, 0, 0.2]),
    #             np.array([ 0.1, 0, 0.2]),
    #             np.array([ 0.0, 0, 0.2]),
    #             np.array([-0.1, 0, 0.2]),
    #             np.array([-0.2, 0, 0.2]),
    #             np.array([-0.3, 0, 0.2])
    #             ]

    # positions =[ # ligne a coté de l'épaule
    #             np.array([-0.15, -0.3, 0.2]),
    #             np.array([-0.15, -0.2, 0.2]),
    #             np.array([-0.15, -0.1, 0.2]),
    #             np.array([-0.15, 0.0, 0.2]),
    #             np.array([-0.15, 0.1, 0.2]),
    #             np.array([-0.15, 0.2, 0.2]),
    #             np.array([-0.15, 0.3, 0.2])
    #             ]

    # positions =[ #vertical line
    #             np.array([ 0.0 , 0, 0.8]),
    #             np.array([ 0.0 , 0, 0.7]),
    #             np.array([ 0.0 , 0, 0.6]),
    #             np.array([ 0.0 , 0, 0.5]),
    #             np.array([ 0.0 , 0, 0.4]),
    #             np.array([ 0.0 , 0, 0.3]),
    #             np.array([ 0.0 , 0, 0.2])
    #             ]

    # MPC ======================================================================

    results_xs = []
    results_us = []
    prim_infeas = []
    dual_infeas = []
    mpc_timer = []

    mpc = MPC(positions, parameters)
    robot_state = mpc.x0

    if not args.no_viz3D:
        viz = Visualization(mpc)

    launch_check = input("Enter to launch") #! fixed with viser PR #614

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

        if not args.no_viz3D:
            viz.update_plot(robot_state[:mpc.n_q], t)
            viz.display_step(xs_no_ee)


        # print(mpc.results.gains.tolist()) # test to get gains

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


    # positions = [np.array([0.5, 0.0, 0.2]), # merry-go-round
    #             np.array([ 0.5, 0.0, 0.5]),
    #             np.array([0.35, 0.35, 0.5]),
    #             np.array([0.35, 0.35, 0.2]),
    #             np.array([0.0, 0.5, 0.2]),
    #             np.array([0.0, 0.5, 0.5]),
    #             np.array([-0.35, 0.35, 0.5]),
    #             np.array([-0.35, 0.35, 0.2]),
    #             np.array([-0.5, 0.0, 0.2]),
    #             np.array([-0.5, 0.0, 0.5]),
    #             np.array([-0.35, -0.35, 0.5]),
    #             np.array([-0.35, -0.35, 0.2]),
    #             np.array([0.0, -0.5, 0.2]),
    #             np.array([0.0, -0.5, 0.5]),
    #             np.array([0.35, -0.35,  0.5]),
    #             np.array([0.35, -0.35,  0.2]),
    #             np.array([0.5, 0.0, 0.2])]
