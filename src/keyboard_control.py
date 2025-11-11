import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from robot import lerobot_IK, lerobot_FK, create_so101, return_jacobian, manipulability
from pynput import keyboard
import threading

np.set_printoptions(linewidth=200)

# Set up the MuJoCo render backend
os.environ["MUJOCO_GL"] = "egl"

# Define joint names
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Absolute path of the XML model
xml_path = "so101/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

# Define joint control increment (in radians)
JOINT_INCREMENT = 0.005
POSITION_INSERMENT = 0.0008

# create robot
robot = create_so101()

# Define joint limits

control_qlimit = [[-2.1, -3.1, -0.0, -1.375,  -1.57, -0.15], 
                  [ 2.1,  0.0,  3.1,  1.475,   3.1,  1.5]]
control_glimit = [[0.125, -0.4,  0.046, -3.1, -0.75, -1.5], 
                  [0.340,  0.4,  0.23, 2.0,  1.57,  1.5]]

# Initialize target joint positions
init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()  # Copy the initial joint positions
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
target_gpos = init_gpos.copy()

# Thread-safe lock
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

# Dictionary to track the currently pressed keys and their direction
keys_pressed = {}

# Handle key press events
def on_press(key):
    try:
        k = key.char.lower()  # Convert to lowercase to handle both upper and lower case inputs
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
        pass  # Handle special keys if necessary

# Handle key release events
def on_release(key):
    try:
        k = key.char.lower()
        if k in keys_pressed:
            with lock:
                del keys_pressed[k]
    except AttributeError:
        pass  # Handle special keys if necessary

# Start the keyboard listener in a separate thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Backup for target_gpos in case of invalid IK
target_gpos_last = init_gpos.copy()
target_qpos_last = init_qpos.copy()

try:
    # Launch the MuJoCo viewer
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        
        start = time.time()
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()
            
            with lock:
                for k, direction in keys_pressed.items():
                    # --- World-frame XY motion (WASD) ---
                    if k in ['w', 's', 'a', 'd']:
                        if k == 'w':
                            move_x, move_y = POSITION_INSERMENT, 0
                        elif k == 's':
                            move_x, move_y = -POSITION_INSERMENT, 0
                        elif k == 'a':
                            move_x, move_y = 0, POSITION_INSERMENT
                        elif k == 'd':
                            move_x, move_y = 0, -POSITION_INSERMENT

                        angle_curr = mjdata.qpos[qpos_indices][0]
                        forward_curr = target_gpos_last[0]
                        x_curr, y_curr = forward_curr * np.cos(angle_curr), forward_curr * np.sin(angle_curr)
                        x_new = x_curr + move_x 
                        y_new = y_curr + move_y

                        theta_update = np.arctan2(y_new,x_new) - np.arctan2(y_curr,x_curr)
                        forward_update = np.sqrt(x_new**2 + y_new**2) - np.sqrt(x_curr**2 + y_curr**2)

                        target_qpos[0] += theta_update
                        target_gpos[0] += forward_update

                        target_qpos[0] = np.clip(target_qpos[0], control_qlimit[0][0], control_qlimit[1][0])
                        target_gpos[0] = np.clip(target_gpos[0], control_glimit[0][0], control_glimit[1][0])

                    # --- Vertical motion (r/f) ---
                    elif k in ['r', 'f']:
                        move_dir = 1 if k == 'r' else -1
                        target_gpos[2] += move_dir * POSITION_INSERMENT
                        target_gpos[2] = np.clip(target_gpos[2], control_glimit[0][2], control_glimit[1][2])

                    # --- Orientation (Roll/Pitch) ---
                    elif k in ['q', 'e', 't', 'g']:
                        ori_idx = {'q': 3, 'e': 3, 't': 4, 'g': 4}[k]
                        move_dir = 1 if k in ['q', 'g'] else -1
                        target_gpos[ori_idx] += move_dir * POSITION_INSERMENT * 4
                        target_gpos[ori_idx] = np.clip(
                            target_gpos[ori_idx],
                            control_glimit[0][ori_idx],
                            control_glimit[1][ori_idx],
                        )

                    # --- Gripper (z/c) ---
                    elif k in ['z', 'c']:
                        move_dir = 1 if k == 'z' else -1
                        target_gpos[5] += move_dir * POSITION_INSERMENT
                        target_gpos[5] = np.clip(target_gpos[5], control_glimit[0][5], control_glimit[1][5])


                               
            fd_qpos = mjdata.qpos[qpos_indices][1:5]
            qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)

            jacobian = return_jacobian(mjdata.qpos[qpos_indices][1:5], robot=robot)

            m_value, condition = manipulability(jacobian)
            print(f'Manipulability: {m_value:.6f}')
            print(f'Condition Number: {condition:.6f}')
            
            if ik_success:  # Check if IK solution is valid
                target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
                mjdata.qpos[qpos_indices] = target_qpos

                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                
                # backup
                target_gpos_last = target_gpos.copy()  # Save backup of target_gpos
            else:
                target_gpos = target_gpos_last.copy()  # Restore the last valid target_gpos

            # Time management to maintain simulation timestep
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("User interrupted the simulation.")
finally:
    listener.stop()  # Stop the keyboard listener
    viewer.close()
