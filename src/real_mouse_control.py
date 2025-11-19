import os
import mujoco
import numpy as np
import time
import threading
from pynput import keyboard, mouse
import rerun as rr
from pathlib import Path

from robot import lerobot_IK, lerobot_FK, create_so101, return_jacobian, manipulability
from lerobot_kinematics import feetech_arm
from urdf_utils import init_rerun_with_urdf, log_joint_angles

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

# ------------------------------
# Setup MuJoCo model and robot
# ------------------------------
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
xml_path = "so101/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
mjdata = mujoco.MjData(mjmodel)

qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
robot = create_so101()

# ------------------------------
# Constants
# ------------------------------
JOINT_INCREMENT = 0.005
POSITION_INCREMENT = 0.0008
MOUSE_SCALE = 0.00025  # Scale factor for mouse movement

control_qlimit = [[-2.1, -3.1, -0.0, -1.375, -1.57, -0.15],
                  [ 2.1,  0.0,  3.1,  1.475,  3.1,  1.5]]
control_glimit = [[0.125, -0.4,  0.046, -3.1, -0.75, -1.5],
                  [0.340,  0.4,  0.23,  2.0,  1.57,  1.5]]

init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
target_gpos = init_gpos.copy()

# Thread lock for keyboard/mouse safety
lock = threading.Lock()

# Mouse tracking
mouse_x, mouse_y = 0, 0
initial_mouse_x, initial_mouse_y = None, None
initial_robot_x, initial_robot_y = None, None
left_mouse_pressed = False

# ------------------------------
# Key Bindings (without WASD and Z/C)
# ------------------------------
key_to_joint_increase = {'r': 2, 'q': 3, 'g': 4}
key_to_joint_decrease = {'f': 2, 'e': 3, 't': 4}
keys_pressed = {}

# ------------------------------
# Mouse Listener
# ------------------------------
def on_move(x, y):
    global mouse_x, mouse_y, initial_mouse_x, initial_mouse_y
    global initial_robot_x, initial_robot_y
    
    with lock:
        # Set initial mouse position on first movement
        if initial_mouse_x is None:
            initial_mouse_x, initial_mouse_y = x, y
            angle_curr = mjdata.qpos[qpos_indices][0]
            forward_curr = init_gpos[0]
            initial_robot_x = forward_curr * np.cos(angle_curr)
            initial_robot_y = forward_curr * np.sin(angle_curr)
        
        mouse_x, mouse_y = x, y

def on_click(x, y, button, pressed):
    global left_mouse_pressed
    
    if button == mouse.Button.left:
        with lock:
            left_mouse_pressed = pressed

mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
mouse_listener.start()

# ------------------------------
# Keyboard Listener
# ------------------------------
def on_press(key):
    try:
        k = key.char.lower()
        if k in key_to_joint_increase:
            with lock:
                keys_pressed[k] = 1
        elif k in key_to_joint_decrease:
            with lock:
                keys_pressed[k] = -1
        elif k == "0":
            with lock:
                global target_qpos, target_gpos, initial_mouse_x, initial_mouse_y
                global initial_robot_x, initial_robot_y
                target_qpos = init_qpos.copy()
                target_gpos = init_gpos.copy()
                # Reset mouse reference
                initial_mouse_x, initial_mouse_y = mouse_x, mouse_y
                angle_curr = init_qpos[0]
                forward_curr = init_gpos[0]
                initial_robot_x = forward_curr * np.cos(angle_curr)
                initial_robot_y = forward_curr * np.sin(angle_curr)
    except AttributeError:
        pass

def on_release(key):
    try:
        k = key.char.lower()
        if k in keys_pressed:
            with lock:
                del keys_pressed[k]
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# ------------------------------
# Connect to the robotic arm motors
# ------------------------------
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
robot_name, joint_paths = init_rerun_with_urdf(f"so101_mouse_rerun_{timestamp}", urdf_path)

# ------------------------------
# Main Loop
# ------------------------------
target_gpos_last = init_gpos.copy()
target_qpos_last = init_qpos.copy()

try:
    start = time.time()
    while time.time() - start < 1000:
        step_start = time.time()

        # ====== Handle mouse input ======
        with lock:
            # Update gripper based on left mouse button (gradual movement)
            if left_mouse_pressed:
                # Close gripper gradually
                target_qpos[5] -= POSITION_INCREMENT * 20
                target_qpos[5] = np.clip(target_qpos[5], control_qlimit[0][5], control_qlimit[1][5])
            else:
                # Open gripper gradually
                target_qpos[5] += POSITION_INCREMENT * 20
                target_qpos[5] = np.clip(target_qpos[5], control_qlimit[0][5], control_qlimit[1][5])
            
            if initial_mouse_x is not None:
                # Calculate mouse delta from initial position
                delta_mouse_y = (mouse_x - initial_mouse_x) * MOUSE_SCALE
                delta_mouse_x = -(mouse_y - initial_mouse_y) * MOUSE_SCALE  # Invert Y for natural control
                
                # Calculate new robot position
                x_new = initial_robot_x + delta_mouse_x
                y_new = initial_robot_y + delta_mouse_y
                
                # Convert to polar coordinates for robot control
                angle_curr = mjdata.qpos[qpos_indices][0]
                forward_curr = target_gpos_last[0]
                x_curr = forward_curr * np.cos(angle_curr)
                y_curr = forward_curr * np.sin(angle_curr)
                
                theta_update = np.arctan2(y_new, x_new) - np.arctan2(y_curr, x_curr)
                forward_update = np.sqrt(x_new**2 + y_new**2) - np.sqrt(x_curr**2 + y_curr**2)
                
                target_qpos[0] = angle_curr + theta_update
                target_gpos[0] = forward_curr + forward_update
                
                target_qpos[0] = np.clip(target_qpos[0], control_qlimit[0][0], control_qlimit[1][0])
                target_gpos[0] = np.clip(target_gpos[0], control_glimit[0][0], control_glimit[1][0])

            # ====== Handle keyboard input (other keys) ======
            for k, direction in keys_pressed.items():
                if k in ['r', 'f']:
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

        # ====== IK, Manipulability ======
        fd_qpos = mjdata.qpos[qpos_indices][1:5]
        qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)
        jacobian = return_jacobian(fd_qpos, robot=robot)
        m_value, condition = manipulability(jacobian)

        # ====== Apply motion ======
        if ik_success:
            target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
            mjdata.qpos[qpos_indices] = target_qpos
            mujoco.mj_step(mjmodel, mjdata)
            follower_arm.action(target_qpos)
            target_gpos_last = target_gpos.copy()
        else:
            target_gpos = target_gpos_last.copy()

        # ====== Rerun Visualization ======
        rr.log("manipulability/value", rr.Scalars(m_value))
        rr.log("condition/kappa", rr.Scalars(condition))

        # Log joint angles for URDF visualization
        joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        log_joint_angles(robot_name, joint_paths, joint_names, target_qpos[:6])
        rr.log("joint_angles", rr.Scalars(target_qpos[:6]))

        # Maintain simulation timing
        time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("User interrupted the simulation.")
finally:
    listener.stop()
    mouse_listener.stop()
    follower_arm.disconnect()