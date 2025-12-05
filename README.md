# Model Predictive Control for AGIMUS WP 6
## Install and Usage
```bash
pixi run main

pixi run main -- <my_args>
```
Options:
- **display**: (default False)
    Displays the 3d visualization of the robot's trajectory
- **debug** : (default False)
    Prints some additional info
- **viz_traj** : (default False)
    Adds the wanted trajectory and actual end effector trajectory waypoints to the 3D visualization
- **perturbate** : (default False)
    Adds perturbation to the warm start (xs and us) of the solver
- **no_joints_lim** : (default False)
    Removes the constraint linked to joint limits (which will no longer be respected)
- **no_torque_lim** : (default False)
    Removes the constraint linked to torque limits (which will no longer be respected)
- **no_orientation_cost** : (default True)
    Removes the constraint linked to orientation of the end effector

## Performance optimization
OpenMP (used by aligator) allows you to parallelize the program on designated threads and/or cores of your CPU: one useful case is prioritizing performance cores of your processor and avoiding economy core. The setup is done in your bash environment with variables.

> [!TIP]
> Get your cpu model : `lscpu | grep -i "model name"`

To display the OpenMP setting upon launch, set:
```bash
export OMP_DISPLAY_ENV=VERBOSE
```

The actual setup is then done with:
```bash
export OMP_PROC_BIND
export OMP_PLACES
export OMP_NUM_THREADS
```
(*"worker threads"* here corresponds to the threads spawned by aligator and *"thread"* refers to the cpu thread)


### `OMP_PLACES` 
Defines the places where the worker threads can be put.  
You can set the places to be threads or cores with :  
```bash
export OMP_PLACES="thread(n)" # The places available will be the nth first consecutive threads.
```

So if you do `export OMP_PLACES="threads(6)` you will see at launch `OMP_PLACES = '{0},{1},{2},{3},{4},{5}'` that lists the first 6 threads where the worker threads can be placed.  
Each `{....}` is a place where one worker thread can be placed. It ignores the repartition of the thread per cores.

```bash
export OMP_PLACES="cores(n)" #The places will be defined as the nth first consecutive cores 
# and will try to put only one worker thread per core.
```

So if you do `export OMP_PLACES="cores(14)` you will see at launch `OMP_PLACES = '{0:2},{2:2},{4:2},{6:2},{8:2},{10:2},{12},{13},{14},{15},{16},{17},{18},{19}'`   
This is done on an i7-13800H which has 6 performances cores with 2 threads each (threads 0 to 11) and 8 economy cores with 1 thread each (threads 12 to 19) so the output is :`{0:2}` = first core starting at thread 0 with threads 0&1 available for one worker thread


### `OMP_PROC_BIND` 
Defines how the worker threads are placed.  
```bash
export OMP_PROC_BIND=SPREAD #  will try to spread the worker threads across the places.  
export OMP_PROC_BIND=CLOSE  # will try to use places near each other.  
export OMP_PROC_BIND=MASTER # will try to keep then close to the master thread.  
```

([more detailed source](https://www.ibm.com/docs/en/xl-fortran-linux/16.1.0?topic=openmp-omp-proc-bind))

### `OMP_NUM_THREADS`
Defines the max number of threads
