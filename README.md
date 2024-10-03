# SEAL (Super Efficient Acados Learning)

## Introduction

SEAL provides tools for learning optimal control policies using learning
methodologies Imitation learning (IL) and Reinforcement Learning (RL) to enhance
Model Predictive Control (MPC) policies. It is built on top of
[acados](https://docs.acados.org/index.html) and [casadi](https://web.casadi.org/).

## Installation

Create python virtual environment and install the (editable) package with the following commands:

``` bash
    sudo pip3 install virtualenv
    cd <PATH_TO_VENV_DIRECTORY>
    virtualenv seal_venv --python=/usr/bin/python3
    source seal_venv/bin/activate
    python -m pip install -e .
```

## Dependencies

SEAL requires the following dependencies:

- [casadi](https://web.casadi.org/) for symbolic computations
- [acados](https://docs.acados.org/index.html) for generating OCP solvers
- [gymnasium](https://gymnasium.farama.org/) for environments.

## Usage

Try an example with the following command:

``` bash
    python examples/linear_system_mpc.py
```