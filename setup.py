from setuptools import setup

setup(
    name="aligator_mpc",
    version="1.0",
    packages=['aligator_mpc'],
    install_requires = ['typed-argument-parser'],
    py_modules=['mpc', 'mpcParameters', 'mpcTrajectoryUtils', 'mpcVisualization', '']
), 