# Model Predictive Control for AGIMUS WP 6
## Install
The code is split in two branches:
- `main` for [aligator 0.15.0](github.com/Simple-Robotics/aligator/releases/tag/v0.15.0)
- `aligator16` for [aligator 0.16.0](https://github.com/Simple-Robotics/aligator/releases/tag/v0.16.0) (branch still in debug)

In both cases the conda environment can be installed with:
```bash
conda env create -f environment.yml
```

## Usage
The main code is located in `mpc_glue.py`, to launch the demo :
```bash
python mpc_glue.py --display
```
Options:
- **display**: (default False)
    Displays the 3d visualisation of the robot's trajectory
- **debug** : (default False)
    Prints some additionnal info
- **viz_traj** : (default False)
    Adds the wanted trajectory and actual end effector trajectory waypoints to the 3D visualisation
- **perturbate** : (default False)
    Adds perturbation to the warm start (xs and us) of the solver
- **no_joints_lim** : (default False)
    Removes the constraint linked to joint limits (which will no longer be respected)
- **no_torque_lim** : (default False)
    Removes the constraint linked to torque limits (which will no longer be respected)
- **no_orientation_cost** : (default True)
    Removes the constraint linked to orientation of the end effector
