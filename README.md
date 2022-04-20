# Intro
Official implementation of **"Learning Forward Dynamics Model and Informed Trajectory Sampler for Safe Quadruped Navigation"** 
*Robotics:Science and Systems (RSS 2022)*

[[Project page](https://awesomericky.github.io/projects/FDM_ITS_navigation/index.html)] [[Paper](https://arxiv.org/abs/2204.08647)]

# Dependencies

Set conda environment
```
conda create -n quadruped_nav python=3.8
conda activate quadruped_nav
```

Install [torch](https://pytorch.org/)(1.10), [numpy](https://numpy.org/install/)(1.21), [matplotlib](https://matplotlib.org/stable/users/getting_started/), [tqdm](https://pypi.org/project/tqdm/), [scipy](https://docs.scipy.org/doc/scipy/getting_started.html#getting-started-ref)

Install required python packages to compute [Dynamic Time Warping](https://dynamictimewarping.github.io/python/) in [Parallel](https://joblib.readthedocs.io/en/latest/installing.html)
```
pip install dtw-python
pip install joblib
```

Install [wandb](https://docs.wandb.ai/quickstart) and login. 'wandb' is a logging system similar to 'tensorboard'.

Install [OMPL](https://ompl.kavrakilab.org/) (Open Motion Planning Library). Python binding version of OMPL is used.
```
Download OMPL installation script in https://ompl.kavrakilab.org/installation.html.
chmod u+x install-ompl-ubuntu.sh
./install-ompl-ubuntu.sh --python
```

# Simulator setup
[RaiSim](https://raisim.com/index.html) is used. Install it following the [installation guide](https://raisim.com/sections/Installation.html).
Then, set up [RaisimGymTorch](https://raisim.com/sections/RaisimGymTorch.html).
```
cd /RAISIM_DIRECTORY_PATH/raisimLib
git clone git@github.com:awesomericky/complex-env-navigation.git
cd complex-env-navigation
python setup.py develop
```

# Path setup
Configure following paths. Parts that should be configured is set with *\TODO: PATH_SETUP_REQUIRED* flag.

1. Trained model weight
    * `cfg['path']['home']` in `/RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/raisimGymTorch/env/envs/test/cfg.yaml`
2. OMPL Python binding
    * `OMPL_PYBIND_PATH` in `/RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/raisimGymTorch/env/envs/train/global_planner.py`

# Quick start 
Run *point-goal navigation* with trained weight
```
python complex-env-navigation/raisimGymTorch/env/envs/test/pgn_runner.py
```

Run *safety-remote control* with trained weight
```
python complex-env-navigation/raisimGymTorch/env/envs/test/src_runner.py
```

# Train model from scratch
Train Forward Dynamics Model
```
python raisimGymTorch/env/envs/train/FDM_train.py -tw /RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/data/command_tracking_flat/final/full_16200.pt
```

Download data to train Informed Trajectory Sampler (386MB)
```

```

Train Informed Trajectory Sampler
```
python raisimGymTorch/env/envs/train/ITS_train.py -fw /RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/data/ITS_train/final/full_450.pt
```







