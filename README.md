# SEAL (Super Efficient Acados Learning)

## Introduction

SEAL provides tools for learning optimal control policies using learning
methodologies Imitation learning (IL) and Reinforcement Learning (RL) to enhance
Model Predictive Control (MPC) policies. It is built on top of
[acados](https://docs.acados.org/index.html) and [casadi](https://web.casadi.org/).

## Installation

### Dependencies

SEAL requires the following dependencies that need to be installed separately:

- [casadi](https://web.casadi.org/) for symbolic computations
- [acados](https://docs.acados.org/index.html) for generating OCP solvers

### Installation steps

Create python virtual environment and install the (editable) package with the following commands:

``` bash
    sudo pip3 install virtualenv
    cd <PATH_TO_VENV_DIRECTORY>
    virtualenv seal_venv --python=/usr/bin/python3
    source seal_venv/bin/activate
```

Install the minimum:

``` bash
    pip install -e .
```

or install with optional dependencies (e.g. for testing):

``` bash
    pip install -e .[test]
```

Install cpu-only pytorch

``` bash
    pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

## Usage

Check out tests for the linear system


``` bash
    python tests/seal/examples/test_linear_system.py
```
