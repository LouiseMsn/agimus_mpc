## Installation:
```bash
mkdir ros2_ws/src && cd ros2_ws/src
git clone git@github.com:LouiseMsn/agimus_mpc.git
cd agimus_mpc
conda env create -f ros2environment.yaml
conda activate aligator_mpc
cd .. #(to src)
vcs import --recursive < go2_mpc_experiments/git-deps.yaml
cd ..

# build eigenpy & coal first
export MAKEFLAGS="-j10" # It is recommended to reduce the number of jobs as you ram might get full easily with the default number.
colcon build --event-handlers console_direct+ --packages-select eigenpy coal --cmake-args \
-DCMAKE_BUILD_TYPE=Release             \
-DCMAKE_PREFIX_PATH=$CONDA_PREFIX      \
-DPYTHON_EXECUTABLE=$(which python)    \
-DCMAKE_CXX_COMPILER_LAUNCHER='ccache' \
-DBUILD_TESTING=OFF                    \
-DBUILD_DOCUMENTATION=OFF              \
-DBUILD_EXAMPLES=OFF                   \
-DBUILD_BENCHMARK=OFF                  \
-DBUILD_BENCHMARKS=OFF                 \
-DBUILD_WITH_COLLISION_SUPPORT=ON      \
-DBUILD_WITH_PROXQP=ON                 \
-DGENERATE_PYTHON_STUBS=OFF            \
-DCOAL_BACKWARD_COMPATIBILITY_WITH_HPP_FCL=ON

source install/setup.bash

# build pinocchio and aligator then
colcon build --event-handlers console_direct+ --packages-select pinocchio aligator  --cmake-args \
-DCMAKE_BUILD_TYPE=Release             \
-DCMAKE_PREFIX_PATH=$CONDA_PREFIX      \
-DPYTHON_EXECUTABLE=$(which python)    \
-DCMAKE_CXX_COMPILER_LAUNCHER='ccache' \
-DBUILD_TESTING=OFF                    \
-DBUILD_DOCUMENTATION=OFF              \
-DBUILD_EXAMPLES=OFF                   \
-DBUILD_BENCHMARK=OFF                  \
-DBUILD_BENCHMARKS=OFF                 \
-DBUILD_WITH_COLLISION_SUPPORT=ON      \
-DBUILD_WITH_PROXQP=ON                 \
-DGENERATE_PYTHON_STUBS=OFF            \
-DCOAL_BACKWARD_COMPATIBILITY_WITH_HPP_FCL=ON


source install/setup.bash

# build pinocchio and aligator then
colcon build --event-handlers console_direct+ --packages-select example-robot-data  --cmake-args \
-DCMAKE_BUILD_TYPE=Release             \
-DCMAKE_PREFIX_PATH=$CONDA_PREFIX      \
-DPYTHON_EXECUTABLE=$(which python)    \
-DCMAKE_CXX_COMPILER_LAUNCHER='ccache' 

source install/setup.bash
```

Add custom mpc package:
```bash
cd ros2_ws/src/agimus_mpc/
pip install -e .
```




