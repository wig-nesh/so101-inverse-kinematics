import os
import mujoco
import numpy as np
import time
import threading
from pathlib import Path
import rerun as rr

from robot import lerobot_IK, lerobot_FK, create_so101, return_jacobian, manipulability
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
# Joint limits matching MuJoCo XML and URDF
control_qlimit = [[-2.2,  -3.14159, 0.0, -2.0, -3.14159, -0.2],
                  [ 2.2,   0.2,     3.14159,  1.8,  3.14159,  2.0]]
control_glimit = [[0.125, -0.4,  0.046, -3.1, -0.75, -1.5],
                  [0.340,  0.4,  0.23,  2.0,  1.57,  1.5]]

# Initialize target joint positions
init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
target_gpos = init_gpos.copy()

# Circular path parameters
CIRCLE_ORIGIN = np.array([0.0, -0.25, 0.08])  # Center of the circle in 3D space (x, y, z)
CIRCLE_RADIUS = 0.05  # Radius of the circle
CIRCLE_PLANE = 'xz'  # Plane: 'xy', 'xz', or 'yz'
CIRCLE_SPEED = 0.01  # Angular speed (radians per step)

# Continuous angle for smooth motion
current_angle = 0.0

# Thread-safe lock
lock = threading.Lock()

# ------------------------------
# Initialize Rerun
# ------------------------------
timestamp = int(time.time())
urdf_path = Path("so101/so101.urdf")
robot_name, joint_paths = init_rerun_with_urdf(f"so101_circular_rerun_{timestamp}", urdf_path)

# ------------------------------
# Main Loop
# ------------------------------
try:
    start = time.time()
    while time.time() - start < 1000:
        step_start = time.time()

        with lock:
            # Compute target position based on current angle
            if CIRCLE_PLANE == 'xy':
                x = CIRCLE_ORIGIN[0] + CIRCLE_RADIUS * np.cos(current_angle)
                y = CIRCLE_ORIGIN[1] + CIRCLE_RADIUS * np.sin(current_angle)
                z = CIRCLE_ORIGIN[2]
            elif CIRCLE_PLANE == 'xz':
                x = CIRCLE_ORIGIN[0] + CIRCLE_RADIUS * np.cos(current_angle)
                y = CIRCLE_ORIGIN[1]
                z = CIRCLE_ORIGIN[2] + CIRCLE_RADIUS * np.sin(current_angle)
            elif CIRCLE_PLANE == 'yz':
                x = CIRCLE_ORIGIN[0]
                y = CIRCLE_ORIGIN[1] + CIRCLE_RADIUS * np.cos(current_angle)
                z = CIRCLE_ORIGIN[2] + CIRCLE_RADIUS * np.sin(current_angle)
            else:
                raise ValueError("Invalid plane. Choose 'xy', 'xz', or 'yz'.")
            
            target_pos = np.array([x, y, z])
            target_qpos[0] = np.arctan2(target_pos[0], -target_pos[1])
            target_gpos[0] = np.sqrt(target_pos[0]**2 + target_pos[1]**2)
            target_gpos[2] = target_pos[2]

            # Increment angle
            current_angle += CIRCLE_SPEED

        # Compute IK
        fd_qpos = mjdata.qpos[qpos_indices][1:5]
        qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)
        
        # Compute manipulability
        jacobian = return_jacobian(fd_qpos, robot=robot)
        m_value, condition = manipulability(jacobian)

        if ik_success:
            target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))
            mjdata.qpos[qpos_indices] = target_qpos
            mujoco.mj_step(mjmodel, mjdata)
            
            # ====== Rerun Visualization ======
            # Log end effector position
            # end_effector_pos = target_pos
            # rr.log("end_effector", rr.Transform3D(translation=end_effector_pos))
            # rr.log("end_effector/point", rr.Points3D([end_effector_pos], radii=0.01, colors=[255, 0, 0]))
            
            # Log circular path
            # rr.log("path/current_angle", rr.Scalars(current_angle))
            # rr.log("path/target_position", rr.Points3D([target_pos], radii=0.005, colors=[0, 255, 0]))
            
            # Log manipulability metrics
            rr.log("manipulability/value", rr.Scalars(m_value))
            rr.log("condition/kappa", rr.Scalars(condition))
            
            # Log joint angles for URDF visualization
            joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
            log_joint_angles(robot_name, joint_paths, joint_names, target_qpos[:6])
            rr.log("joint_angles", rr.Scalars(target_qpos[:6]))
        else:
            print(f"IK failed for angle: {current_angle:.3f}")

        # Time management to maintain simulation timestep
        time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("User interrupted the simulation.")
