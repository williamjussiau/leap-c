# leap-c (Learning Predictive Control)

## Introduction

leap-c provides tools for learning optimal control policies using learning
methodologies Imitation learning (IL) and Reinforcement Learning (RL) to enhance
Model Predictive Control (MPC) policies. It is built on top of
[acados](https://docs.acados.org/index.html) and [casadi](https://web.casadi.org/).

## Installation

### Dependencies

leap-c requires the following dependencies that need to be installed separately:

- [casadi](https://web.casadi.org/) for symbolic computations
- [acados](https://docs.acados.org/index.html) for generating OCP solvers

### acados as a submodule

leap-c uses acados as a submodule. To clone the repository with the submodule, use the following command:

``` bash
    git clone --recurse-submodules git@github.com:leap-c/leap-c.git
```

Follow the installation instructions for acados [here](https://docs.acados.org/installation/)

### Installation steps

Create python virtual environment and install the (editable) package with the following commands:

``` bash
    sudo pip3 install virtualenv
    cd <PATH_TO_VENV_DIRECTORY>
    virtualenv .venv --python=/usr/bin/python3
    source .venv/bin/activate
```

Install the required casadi version with the installation script:

``` bash
    cd <PATH_TO_LEAP_C_DIRECTORY>
    ./install_new_casadi_py311_x86_64.sh
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
    python tests/leap_c/examples/test_linear_system.py
```

## Questions/ Contact

Write to dirk.p.reinhardt@ntnu.no
