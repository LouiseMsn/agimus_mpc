from setuptools import setup

setup(
    name="aligator_mpc",
    version="1.0",
    packages=['aligator_mpc'],
    install_requires = [
                        'viser',
                        'pydantic'
                        ],
    py_modules=['mpc', 'mpcParameters', 'mpcTrajectoryUtils', '']
), 