# mujoco dynamics docking (sdf version)

Welcome to the mujodo dynamics docking (sdf version) repository! 

This project is designed to provide a comprehensive solution for trajectory planning and dynamics simulation in robotics using the Mujoco simulator. It's built on top of the powerful Pinocchio library, which allows for efficient forward kinematics calculations.


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This is a repository for mujoco dynamics simulation and compliance control.

## Features

- signed distance field (sdf)
- dynamics simulation
- compliance control

## Usage

To use this repository, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/ming751/rm_robot.git
   ```

   ```bash
   cd rm_robot
   ```

2. Install the dependencies(if you use dymamic simulation, you need to install pinocchio,pinocchio must be installed in linux system):

   1) create an conda environment and activate it:
   ```bash
   conda create -n pin_mjcf python=3.10
   conda activate pin_mjcf
   ```
   
   2) install the pinocchio dependencies:
   ```bash
   pip install pin
   ```
   
   3) install other dependencies:
   ```bash
   pip install -r requirement.txt
   ```

3. Run the script:

   ### Main simulation script

   ```bash
   python task_dynamics_mujoco_control.py
   ```

## Contributing

We welcome contributions to the RM Robot project! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b my-feature-branch
   ```
3. Make your changes and commit them:
   ```bash
   git commit -am 'Add new feature'
   ```
4. Push your changes to your fork:
   ```bash
   git push origin my-feature-branch
   ```
5. Create a pull request on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.