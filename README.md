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

Install [torch](https://pytorch.org/)(1.10.1), [numpy](https://numpy.org/install/)(1.21.2), [matplotlib](https://matplotlib.org/stable/users/getting_started/), [scipy](https://docs.scipy.org/doc/scipy/getting_started.html#getting-started-ref), [ruamel.yaml](https://pypi.org/project/ruamel.yaml/)
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install numpy=1.21.2
conda install matplotlib
conda install scipy
pip install ruamel.yaml
```

Install [wandb](https://docs.wandb.ai/quickstart) and login. 'wandb' is a logging system similar to 'tensorboard'.
```
pip install wandb
wandb login
```

Install required python packages to compute [Dynamic Time Warping](https://dynamictimewarping.github.io/python/) in [Parallel](https://joblib.readthedocs.io/en/latest/installing.html)
```
pip install dtw-python
pip install fastdtw
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

1. Project directory
    * `cfg['path']['home']` in `/RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/raisimGymTorch/env/envs/test/cfg.yaml`
2. OMPL Python binding
    * `OMPL_PYBIND_PATH` in `/RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/raisimGymTorch/env/envs/train/global_planner.py`

# Train model
Set `logging: True` in `/RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/raisimGymTorch/env/envs/train/cfg.yaml`, if you want to enable wandb logging.

Train Forward Dynamics Model (FDM).
* Click 'c' to continue when pdb stops the code
* To quit the training, click 'Ctrl + c' to call pdb. Then click 'q'.
* Path of the trained velocity command tracking controller should be given with `-tw` flag. 
* Evaluations of FDM are visualized in `/RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/trajectory_prediction_plot`.
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
* Click 'c' to continue when pdb stops the code. 
* To quit the training, click 'Ctrl + c' to call pdb. Then click 'q'.
* Path of the trained Forward Dynamics Model should be given with `-fw` flag.
```
python raisimGymTorch/env/envs/train/ITS_train.py -fw /RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/data/FDM_train/XXX/full_XXX.pt
```

# Run demo
Configure the trained weight paths (`cfg['path']['FDM']` and `cfg['path']['ITS']`) in `/RAISIM_DIRECTORY_PATH/raisimLib/complex-env-navigation/raisimGymTorch/env/envs/test/cfg.yaml`. 
Parts that should be configured is set with `TODO: WEIGHT_PATH_SETUP_REQUIRED` flag.

Open [RaiSim Unity](https://raisim.com/sections/RaisimUnity.html) to see the visualized simulation.

Run *point-goal navigation* with trained weight (click 'c' to continue when pdb stops the code)
```
python raisimGymTorch/env/envs/test/pgn_runner.py
```

Run *safety-remote control* with trained weight (click 'c' to continue when pdb stops the code)
```
python raisimGymTorch/env/envs/test/src_runner.py
```
To quit running the demo, click 'Ctrl + c' to call pdb. Then click 'q'.

# Extra notes
* **This repository is not maintained anymore.** If you have a question, send an email to awesomericky@kaist.ac.kr.
* We don't take questions regarding installation. If you install the dependencies successfully, you can easily run this.
* For the codes in rsc/, ANYbotics' license is applied. MIT license otherwise.
* More details of the provided velocity command tracking controller for quadruped robots in flat terrain can be found in this [paper](https://arxiv.org/abs/1901.08652) and [repository](https://github.com/awesomericky/velocity-command-tracking-controller-for-quadruped-robot).

# Cite
```
@INPROCEEDINGS{Kim-RSS-22, 
    AUTHOR    = {Yunho Kim AND Chanyoung Kim AND Jemin Hwangbo}, 
    TITLE     = {{Learning Forward Dynamics Model and Informed Trajectory Sampler for Safe Quadruped Navigation}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2022}, 
    ADDRESS   = {New York City, NY, USA}, 
    MONTH     = {June}, 
    DOI       = {10.15607/RSS.2022.XVIII.069} 
}
```







