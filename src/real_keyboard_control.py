# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import os
import mujoco
import numpy as np
import time
from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot, feetech_arm

from pynput import keyboard
import threading
from pathlib import Path
import rerun as rr

# For Feetech Motors
from lerobot_kinematics.lerobot.feetech import FeetechMotorsBus
import json

from urdf_utils import init_rerun_with_urdf, log_joint_angles
from robot import return_jacobian, manipulability, create_so101

np.set_printoptions(linewidth=200)

# Define joint names
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Absolute path of the XML model
xml_path = "so101/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

robot = get_robot('so100')
robot_ik = create_so101()  # Robot for manipulability calculations

# Define joint control increment (in radians)
JOINT_INCREMENT = 0.005*1.5
POSITION_INCREMENT = 0.0008*1.5

# Define joint limits matching MuJoCo XML and URDF
# [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
control_qlimit = [[-2.2,  -3.14159, 0.0, -2.0, -3.14159, -0.2],
                  [ 2.2,   0.2,     3.14159,  1.8,  3.14159,  2.0]]
control_glimit = [[0.125, -0.4,  0.046, -3.1, -0.75, -1.5],
                  [0.340,  0.4,  0.23,  2.0,  1.57,  1.5]]

# Initialize target joint positions
init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()
init_gpos = lerobot_FK(init_qpos[1:5],robot=robot)
target_gpos = init_gpos.copy()

# Thread-safe lock for key press management
lock = threading.Lock()

# Define key mappings
key_to_joint_increase = {
    'w': 0,  # Move forward
    'a': 1,  # Move right
    'r': 2,  # Move up
    'q': 3,  # Roll +
    'g': 4,  # Pitch +
    'z': 5,  # Gripper +
}

key_to_joint_decrease = {
    's': 0,  # Move backward
    'd': 1,  # Move left
    'f': 2,  # Move down
    'e': 3,  # Roll -
    't': 4,  # Pitch -
    'c': 5,  # Gripper -
}

# Dictionary to track currently pressed keys and their direction
keys_pressed = {}

# Callback for key press events
def on_press(key):
    try:
        k = key.char.lower()  # Convert to lowercase to handle both lowercase and uppercase inputs
        if k in key_to_joint_increase:
            with lock:
                keys_pressed[k] = 1  # Increase direction
        elif k in key_to_joint_decrease:
            with lock:
                keys_pressed[k] = -1  # Decrease direction
        elif k == "0":
            with lock:
                global target_qpos, target_gpos
                target_qpos = init_qpos.copy()  # Reset to initial position
                target_gpos = init_gpos.copy()  # Reset to initial gripper position
        print(f'{key}')

    except AttributeError:
        pass  # Handle special keys if needed

# Callback for key release events
def on_release(key):
    try:
        k = key.char.lower()
        if k in keys_pressed:
            with lock:
                del keys_pressed[k]
    except AttributeError:
        pass  # Handle special keys if needed

# Start the keyboard listener in a separate thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

target_gpos_last = init_gpos.copy()

# Connect to the robotic arm motors
motors = {"shoulder_pan": (1, "sts3215"),
          "shoulder_lift": (2, "sts3215"),
          "elbow_flex": (3, "sts3215"),
          "wrist_flex": (4, "sts3215"),
          "wrist_roll": (5, "sts3215"),
          "gripper": (6, "sts3215")}

follower_arm = feetech_arm(driver_port="/dev/ttyACM0", calibration_file="so101/calibration.json" )

# ------------------------------
# Initialize Rerun
# ------------------------------
timestamp = int(time.time())
urdf_path = Path("so101/so101.urdf")
robot_name, joint_paths = init_rerun_with_urdf(f"so101_real_control_{timestamp}", urdf_path)

t = 0
start = time.time()

try:
    # Initialize robot position
    if t == 0:
        mjdata.qpos[qpos_indices] = init_qpos
        mujoco.mj_step(mjmodel, mjdata)
        mjdata.qpos[qpos_indices] = init_qpos   
        mujoco.mj_step(mjmodel, mjdata)    
    t = t + 1
    
    while time.time() - start < 1000:
        step_start = time.time()

        with lock:
                # for k, direction in keys_pressed.items():
                #     if k in key_to_joint_increase:
                #         position_idx = key_to_joint_increase[k]
                #         if position_idx == 1 or position_idx == 5:
                #             position_idx = 0 if position_idx == 1 else 5
                #             if target_qpos[position_idx] < control_qlimit[1][position_idx] - JOINT_INCREMENT * direction:
                #                 target_qpos[position_idx] += JOINT_INCREMENT * direction
                #         elif position_idx == 4 or position_idx == 3:
                #             if target_gpos[position_idx] <= control_glimit[1][position_idx]:
                #                 target_gpos[position_idx] += POSITION_INCREMENT * direction * 4
                #         else:
                #             if target_gpos[position_idx] <= control_glimit[1][position_idx]:
                #                 target_gpos[position_idx] += POSITION_INCREMENT * direction

                #     elif k in key_to_joint_decrease:
                #         position_idx = key_to_joint_decrease[k]
                #         if position_idx == 1 or position_idx == 5:
                #             position_idx = 0 if position_idx == 1 else 5
                #             if target_qpos[position_idx] > control_qlimit[0][position_idx] - JOINT_INCREMENT * direction:
                #                 target_qpos[position_idx] += JOINT_INCREMENT * direction
                #         elif position_idx == 4 or position_idx == 3:
                #             if target_gpos[position_idx] >= control_glimit[0][position_idx]:
                #                 target_gpos[position_idx] += POSITION_INCREMENT * direction * 4
                #         else:
                #             if target_gpos[position_idx] >= control_glimit[0][position_idx]:
                #                 target_gpos[position_idx] += POSITION_INCREMENT * direction

                for k, direction in keys_pressed.items():
                    if k in ['w', 's', 'a', 'd']:
                        if k == 'w':
                            move_x, move_y = POSITION_INCREMENT, 0
                        elif k == 's':
                            move_x, move_y = -POSITION_INCREMENT, 0
                        elif k == 'd':
                            move_x, move_y = 0, POSITION_INCREMENT
                        elif k == 'a':
                            move_x, move_y = 0, -POSITION_INCREMENT

                        angle_curr = mjdata.qpos[qpos_indices][0]
                        forward_curr = target_gpos_last[0]
                        x_curr, y_curr = forward_curr * np.cos(angle_curr), forward_curr * np.sin(angle_curr)
                        x_new = x_curr + move_x
                        y_new = y_curr + move_y

                        theta_update = np.arctan2(y_new, x_new) - np.arctan2(y_curr, x_curr)
                        forward_update = np.sqrt(x_new**2 + y_new**2) - np.sqrt(x_curr**2 + y_curr**2)

                        target_qpos[0] += theta_update
                        target_gpos[0] += forward_update

                        target_qpos[0] = np.clip(target_qpos[0], control_qlimit[0][0], control_qlimit[1][0])
                        target_gpos[0] = np.clip(target_gpos[0], control_glimit[0][0], control_glimit[1][0])

                    elif k in ['r', 'f']:
                        move_dir = 1 if k == 'r' else -1
                        target_gpos[2] += move_dir * POSITION_INCREMENT
                        target_gpos[2] = np.clip(target_gpos[2], control_glimit[0][2], control_glimit[1][2])

                    elif k in ['q', 'e', 't', 'g']:
                        ori_idx = {'q': 3, 'e': 3, 't': 4, 'g': 4}[k]
                        move_dir = 1 if k in ['q', 'g'] else -1
                        target_gpos[ori_idx] += move_dir * POSITION_INCREMENT * 4
                        target_gpos[ori_idx] = np.clip(
                            target_gpos[ori_idx],
                            control_glimit[0][ori_idx],
                            control_glimit[1][ori_idx],
                        )

                    elif k in ['z', 'c']:
                        move_dir = 1 if k == 'z' else -1
                        target_qpos[5] += move_dir * POSITION_INCREMENT * 4
                        target_qpos[5] = np.clip(target_qpos[5], control_qlimit[0][5], control_qlimit[1][5])

        # print("target_gpos:", [f"{x:.3f}" for x in target_gpos])
        # fd_qpos = np.concatenate(([0.0,], mjdata.qpos[qpos_indices][1:5]))
        fd_qpos = mjdata.qpos[qpos_indices][1:5]
        qpos_inv, IK_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)
        
        # Compute manipulability for visualization
        jacobian = return_jacobian(fd_qpos, robot=robot_ik)
        m_value, condition = manipulability(jacobian)

        if np.all(qpos_inv != -1.0):  # Check if IK solution is valid
            target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
            print("target_qpos:", [f"{x:.3f}" for x in target_qpos])
            mjdata.qpos[qpos_indices] = target_qpos
            # mjdata.ctrl[qpos_indices] = target_qpos
            
            mujoco.mj_step(mjmodel, mjdata)

            follower_arm.action(target_qpos)
            target_gpos_last = target_gpos.copy()
            
            # ====== Rerun Visualization ======
            # Log end effector position
            # angle_curr = target_qpos[0]
            # forward_curr = target_gpos[0]
            # x_pos = forward_curr * np.cos(angle_curr)
            # y_pos = forward_curr * np.sin(angle_curr)
            # z_pos = target_gpos[2]
            # end_effector_pos = np.array([x_pos, y_pos, z_pos])
            # rr.log("end_effector", rr.Transform3D(translation=end_effector_pos))
            # rr.log("end_effector/point", rr.Points3D([end_effector_pos], radii=0.01, colors=[255, 0, 0]))
            
            # Log manipulability metrics
            rr.log("manipulability/value", rr.Scalars(m_value))
            rr.log("condition/kappa", rr.Scalars(condition))
            
            # Log joint angles for URDF visualization
            joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
            log_joint_angles(robot_name, joint_paths, joint_names, target_qpos[:6])
            rr.log("joint_angles", rr.Scalars(target_qpos[:6]))
        else:
            target_gpos = target_gpos_last.copy()

        # print()
        time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("User interrupted the simulation.")
finally:
    listener.stop()
    follower_arm.disconnect()
