# Intro
Official implementation of **"Learning Forward Dynamics Model and Informed Trajectory Sampler for Safe Quadruped Navigation"** 
*Robotics:Science and Systems (RSS 2022)*

[[Project page](https://awesomericky.github.io/projects/FDM_ITS_navigation/index.html)] [[Paper](https://arxiv.org/abs/2204.08647)]

<img width=700 src='demo.gif'>

# Dependencies

Set conda environment
```
conda create -n quadruped_nav python=3.8
conda activate quadruped_nav
```

Install [torch](https://pytorch.org/)(1.10), [numpy](https://numpy.org/install/)(1.21), [matplotlib](https://matplotlib.org/stable/users/getting_started/), [tqdm](https://pypi.org/project/tqdm/), [scipy](https://docs.scipy.org/doc/scipy/getting_started.html#getting-started-ref)

Install [wandb](https://docs.wandb.ai/quickstart) and login. 'wandb' is a logging system similar to 'tensorboard'.

Install required python packages to compute [Dynamic Time Warping](https://dynamictimewarping.github.io/python/) in [Parallel](https://joblib.readthedocs.io/en/latest/installing.html)
```
pip install dtw-python
pip install joblib
```

Install [OMPL](https://ompl.kavrakilab.org/) (Open Motion Planning Library). Python binding version of OMPL is used.
```
Download OMPL installation script in https://ompl.kavrakilab.org/installation.html.
chmod u+x install-ompl-ubuntu.sh
./install-ompl-ubuntu.sh --python
```

# Simulator setup
[RaiSim](https://raisim.com/index.html) is used. Install it following the [installation guide](https://raisim.com/sections/Installation.html).

Then, set up [RaisimGymTorch](https://raisim.com/sections/RaisimGymTorch.html) as following.
```
cd /RAISIM_DIRECTORY_PATH/raisimLib
git clone git@github.com:awesomericky/complex-env-navigation.git
cd complex-env-navigation
python setup.py develop
```

# Path setup
Configure following paths. Parts that should be configured is set with `TODO: PATH_SETUP_REQUIRED` flag.

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
Set `logging: True` in `/RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/raisimGymTorch/env/envs/train/cfg.yaml` to enable wandb logging.

Train Forward Dynamics Model (FDM)
```
python raisimGymTorch/env/envs/train/FDM_train.py -tw /RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/data/command_tracking_flat/final/full_16200.pt
```

Download data to train Informed Trajectory Sampler (386MB) [[link](https://drive.google.com/file/d/1R7EyMPIyNkHme9H-z20VN1BkFeDVS4an/view?usp=sharing)]
```
# Unzip the downloaded zip file and move it to required path.
unzip analytic_planner_data.zip
mv analytic_planner_data /RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/.
```

Train Informed Trajectory Sampler (ITS)
```
python raisimGymTorch/env/envs/train/ITS_train.py -fw /RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/data/FDM_train/final/full_1500.pt
```

# Etc
More details of the provided velocity command tracking controller for quadruped robots in flat terrain can be found in this [paper](https://arxiv.org/abs/1901.08652) and [repository](https://github.com/awesomericky/velocity-command-tracking-controller-for-quadruped-robot).

# Cite
```
@INPROCEEDINGS{Kim-RSS-22, 
    AUTHOR    = {Yunho Kim AND Chanyoung Kim AND Jemin Hwangbo}, 
    TITLE     = {Learning Forward Dynamics Model and Informed Trajectory Sampler for Safe Quadruped Navigation}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2022}, 
    ADDRESS   = {New York, USA}, 
    MONTH     = {June}
} 
```







