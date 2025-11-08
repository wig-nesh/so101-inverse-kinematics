import mujoco
import mujoco.viewer
import numpy as np

# Load model and data
m = mujoco.MjModel.from_xml_path('so101/so101.mujoco.xml')
d = mujoco.MjData(m)

# Number of floating base qpos elements (7: x,y,z + quaternion)
FREEJOINT_OFFSET = 7

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():

        d.qpos[FREEJOINT_OFFSET:] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Update forward kinematics
        mujoco.mj_forward(m, d)
        # Sync viewer
        viewer.sync()