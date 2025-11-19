# SO101 Inverse Kinematics

A comprehensive inverse kinematics implementation for the SO101 robotic arm, featuring workspace analysis, multiple control interfaces, and real-time visualization.

## Overview

This project implements forward and inverse kinematics for the SO101 robotic arm using differential kinematics methods. It provides various control interfaces including keyboard, mouse, and hand tracking, along with tools for workspace analysis and visualization. The slides mention the limitations and why they had to be taken.

## Features

- **Forward and Inverse Kinematics**: Efficient implementations using differential kinematics
- **Multiple Control Interfaces**:
  - Keyboard control (with Wayland support)
  - Mouse control
  - Hand tracking
  - Circular path following
- **Workspace Analysis**: Compute and visualize the robot's reachable workspace
- **Real-time Visualization**: Integration with Rerun for interactive visualization
- **Hardware Support**: Direct control of Feetech servos for physical robot deployment

## Installation

### Prerequisites

- Python >=3.10
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/wig-nesh/so101-inverse-kinematics.git
cd so101-inverse-kinematics
```

2. Install dependencies:
```bash
cd src
uv pip install -e .
```
The `so101/calibration.json` is taken from the calibration run using Hugging Face's open source code, and is needed only when using the actual robot motors. If you have the arm you know what this json is :) 

## Usage

All scripts should be run from the project root directory using the format:

```bash
python src/<script_name>.py
```

### Available Scripts

#### Simulation Control

- **Keyboard Control**:
  ```bash
  python src/keyboard_control.py
  # For Wayland systems:
  python src/keyboard_control_rerun_wayland.py
  ```

- **Mouse Control**:
  ```bash
  python src/mouse_control.py
  ```

- **Circular Path Following**:
  ```bash
  python src/circular_path.py
  # With Rerun visualization:
  python src/circular_path_rerun.py
  ```

- **Hand Tracking**:
  ```bash
  python src/hand_tracking.py
  ```

#### Real Hardware Control

- **Real Robot Keyboard Control**:
  ```bash
  python src/real_keyboard_control.py
  ```

- **Real Robot Mouse Control**:
  ```bash
  python src/real_mouse_control.py
  ```

- **Real Robot Circular Path**:
  ```bash
  python src/real_circular_path.py
  ```

#### Analysis Tools

- **Workspace Analysis**:
  ```bash
  python src/workspace_analysis.py
  ```

- **Feasibility Projection** (and its failure):
  ```bash
  python src/feasibility_projection.py
  ```

## Credits and Acknowledgments

This project builds upon and is inspired by several excellent open-source projects and educational resources:

- **[LeRobot](https://github.com/huggingface/lerobot)** by Hugging Face - For the SO100/SO101 robot platform and community
- **[lerobot-kinematics](https://github.com/your-repo-here)** - Base kinematics library implementation
- **Peter Corke and Jesse Haviland** - For their excellent [tutorial on manipulator differential kinematics](https://github.com/petercorke/robotics-toolbox-python) and the ET format