## Installation

1. Clone the repository:
```sh
    git clone git@github.com:leonthorm/thesis.git
    git submodule sync
    git submodule update --init --recursive
```
2. Build the python bindings of the crazyflie-firmware
```sh
    cd deps/crazyflie-firmware
    make cf2_defconfig
    make bindings_python
    export PYTHONPATH=path/to/thesis/deps/crazyflie-firmware:$PYTHONPATH
```
## Usage

To run the simulation, execute the following command:
```sh
    python3 scripts/run_dagger_coltrans.py
```

## research
[IL research](https://docs.google.com/document/d/1qL__5ltoS9RlNtAtyIXOkichVQ9TchGlXO6cpNntQVI/edit?usp=sharing)
