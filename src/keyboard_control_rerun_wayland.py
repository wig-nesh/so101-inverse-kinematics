import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr

from robot import create_so101, lerobot_FK, lerobot_IK, manipulability, return_jacobian
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
qpos_indices = np.array(
    [mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES]
)

robot = create_so101()

# ------------------------------
# Constants
# ------------------------------
# Note: Terminal repeat rates vary. If the robot moves too slow/fast,
# adjust these increments or your terminal's 'xset r rate' (on X11) / hyprland config.
JOINT_INCREMENT = 0.005
POSITION_INCREMENT = 0.0008

control_qlimit = [
    [-2.1, -3.1, -0.0, -1.375, -1.57, -0.15],
    [2.1, 0.0, 3.1, 1.475, 3.1, 1.5],
]
control_glimit = [
    [0.125, -0.4, 0.046, -3.1, -0.75, -1.5],
    [0.340, 0.4, 0.23, 2.0, 1.57, 1.5],
]

init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
target_gpos = init_gpos.copy()


# ------------------------------
# Terminal Input Helper
# ------------------------------
class KeyPoller:
    def __enter__(self):
        # Save the terminal settings
        self.fd = sys.stdin.fileno()
        self.new_term = termios.tcgetattr(self.fd)
        self.old_term = termios.tcgetattr(self.fd)
        # New terminal setting unbuffered
        self.new_term[3] = self.new_term[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(self.fd, termios.TCSANOW, self.new_term)
        return self

    def __exit__(self, type, value, traceback):
        # Restore terminal settings
        termios.tcsetattr(self.fd, termios.TCSANOW, self.old_term)

    def get_chars(self):
        # Read all characters currently in buffer (non-blocking)
        chars = []
        while select.select([sys.stdin], [], [], 0)[0]:
            chars.append(sys.stdin.read(1))
        return chars


# Key Logic:
# Map keys to directions
key_to_joint_increase = {"w": 0, "a": 1, "r": 2, "q": 3, "g": 4, "z": 5}
key_to_joint_decrease = {"s": 0, "d": 1, "f": 2, "e": 3, "t": 4, "c": 5}

# To simulate "holding" a key in a terminal, we track the last time we saw it.
# If we saw it recently (< KEY_DECAY seconds ago), we consider it pressed.
keys_last_seen = {}
KEY_DECAY = 0.08  # Seconds. Tuned for standard keyboard repeat rates.

# Initialize Rerun
timestamp = int(time.time())
urdf_path = Path("so101/so101.urdf")
robot_name, joint_paths = init_rerun_with_urdf(
    f"so101_keyboard_rerun_{timestamp}", urdf_path
)

# Main Loop
target_gpos_last = init_gpos.copy()
target_qpos_last = init_qpos.copy()

print(f"Starting simulation. Click this terminal to control.")
print(f"Controls: WASD (Planar), RF (Vert), QE/TG (Orient), ZC (Grip), 0 (Reset)")
print(f"Press 'Ctrl+C' or 'Esc' to exit.")

try:
    # Wrap the loop in the KeyPoller context manager to handle raw input
    with KeyPoller() as key_poller:
        start = time.time()
        while time.time() - start < 1000:
            step_start = time.time()

            # 1. Process Input
            chars = key_poller.get_chars()
            current_time = time.time()

            for char in chars:
                k = char.lower()
                if k == "\x1b":  # ESC to exit
                    raise KeyboardInterrupt
                keys_last_seen[k] = current_time

                # Handle single-trigger reset
                if k == "0":
                    target_qpos = init_qpos.copy()
                    target_gpos = init_gpos.copy()

            # Determine which keys are currently "active" (held down)
            # We check all mapped keys to see if their timestamp is fresh
            active_keys = {}

            for k, idx in key_to_joint_increase.items():
                if current_time - keys_last_seen.get(k, 0.0) < KEY_DECAY:
                    active_keys[k] = 1  # Direction 1

            for k, idx in key_to_joint_decrease.items():
                if current_time - keys_last_seen.get(k, 0.0) < KEY_DECAY:
                    active_keys[k] = -1  # Direction -1

            # 2. Apply Control Logic
            if active_keys:
                # Iterate through our active inputs and apply updates
                # We iterate active_keys so simultaneous presses (if caught) work
                for k, direction in active_keys.items():
                    # World-frame XY motion (WASD)
                    if k in ["w", "s", "a", "d"]:
                        if k == "w":
                            move_x, move_y = POSITION_INCREMENT, 0
                        elif k == "s":
                            move_x, move_y = -POSITION_INCREMENT, 0
                        elif k == "a":
                            move_x, move_y = 0, POSITION_INCREMENT
                        elif k == "d":
                            move_x, move_y = 0, -POSITION_INCREMENT

                        angle_curr = mjdata.qpos[qpos_indices][0]
                        # Use target_gpos_last to prevent drift feedback loop
                        forward_curr = target_gpos_last[0]

                        x_curr = forward_curr * np.cos(angle_curr)
                        y_curr = forward_curr * np.sin(angle_curr)

                        x_new = x_curr + move_x
                        y_new = y_curr + move_y

                        theta_update = np.arctan2(y_new, x_new) - np.arctan2(
                            y_curr, x_curr
                        )
                        forward_update = np.sqrt(x_new**2 + y_new**2) - np.sqrt(
                            x_curr**2 + y_curr**2
                        )

                        target_qpos[0] += theta_update
                        target_gpos[0] += forward_update

                        target_qpos[0] = np.clip(
                            target_qpos[0], control_qlimit[0][0], control_qlimit[1][0]
                        )
                        target_gpos[0] = np.clip(
                            target_gpos[0], control_glimit[0][0], control_glimit[1][0]
                        )

                    # Vertical motion (r/f)
                    elif k in ["r", "f"]:
                        # r is mapped to increase (up), f to decrease (down)
                        # direction is already +1 or -1 based on the dict lookup above
                        target_gpos[2] += direction * POSITION_INCREMENT
                        target_gpos[2] = np.clip(
                            target_gpos[2], control_glimit[0][2], control_glimit[1][2]
                        )

                    # Orientation (Roll/Pitch)
                    elif k in ["q", "e", "t", "g"]:
                        ori_idx = {"q": 3, "e": 3, "t": 4, "g": 4}[k]
                        # For q/g direction is 1, for e/t direction is -1 (handled by active_keys logic)
                        target_gpos[ori_idx] += direction * POSITION_INCREMENT * 4
                        target_gpos[ori_idx] = np.clip(
                            target_gpos[ori_idx],
                            control_glimit[0][ori_idx],
                            control_glimit[1][ori_idx],
                        )

                    # Gripper (z/c)
                    elif k in ["z", "c"]:
                        target_qpos[5] += direction * POSITION_INCREMENT
                        target_qpos[5] = np.clip(
                            target_qpos[5], control_qlimit[0][5], control_qlimit[1][5]
                        )

            # 3. IK and Physics
            fd_qpos = mjdata.qpos[qpos_indices][1:5]
            qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)

            jacobian = return_jacobian(fd_qpos, robot=robot)
            m_value, condition = manipulability(jacobian)

            if ik_success:
                target_qpos = np.concatenate(
                    (target_qpos[0:1], qpos_inv[:4], target_qpos[5:])
                )
                mjdata.qpos[qpos_indices] = target_qpos
                mujoco.mj_step(mjmodel, mjdata)
                target_gpos_last = target_gpos.copy()
            else:
                target_gpos = target_gpos_last.copy()

            # 4. Rerun Logging
            # Color mapping for manipulability (Green=Good, Red=Bad)
            # Simple lerp: max_m ~ 0.08 for this robot?
            m_norm = np.clip(m_value / 0.08, 0, 1)
            color = [int(255 * (1 - m_norm)), int(255 * m_norm), 0]

            # Compute EE position for viz
            ee_x = target_gpos[0] * np.cos(target_qpos[0])
            ee_y = target_gpos[0] * np.sin(target_qpos[0])
            ee_z = target_gpos[2]
            ee_pos = np.array([ee_x, ee_y, ee_z])

            # Log trace
            rr.log(
                "end_effector/trace", rr.Points3D([ee_pos], radii=0.005, colors=color)
            )

            # Log metrics
            rr.log("manipulability/value", rr.Scalars(m_value))
            rr.log("condition/kappa", rr.Scalars(condition))

            # Log joint angles
            joint_names = [
                "Rotation",
                "Pitch",
                "Elbow",
                "Wrist_Pitch",
                "Wrist_Roll",
                "Jaw",
            ]
            log_joint_angles(robot_name, joint_paths, joint_names, target_qpos[:6])
            rr.log("joint_angles", rr.Scalars(target_qpos[:6]))

            # 5. Timing
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("\nUser interrupted the simulation.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    # Termios settings are restored by the context manager's __exit__
    print("Exiting.")
