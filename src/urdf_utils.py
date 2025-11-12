"""
Utility functions for working with URDF files in Rerun visualization.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple
import rerun as rr
from rerun import RotationAxisAngle


def build_joint_paths(urdf_path: Path) -> Tuple[str, Dict[str, Dict[str, any]]]:
    """
    Build hierarchical joint paths from URDF structure.
    
    This creates paths in the format parent/joint/child which matches how Rerun's
    URDF loader creates the entity tree.
    
    Args:
        urdf_path: Path to the URDF file
        
    Returns:
        Tuple of (robot_name, joint_paths_dict) where joint_paths_dict maps
        joint_name -> {'path': hierarchical_path, 'axis': rotation_axis}
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    robot_name = root.get('name', 'robot')
    
    # Map joint name -> (parent, child, axis)
    joint_info = {}
    for joint in root.findall('.//joint[@type="revolute"]'):
        name = joint.get('name')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        axis_elem = joint.find('axis')
        axis = [float(x) for x in axis_elem.get('xyz').split()] if axis_elem is not None else [0, 0, 1]
        joint_info[name] = {'parent': parent, 'child': child, 'axis': axis}
    
    # Build hierarchical paths: parent/joint/child
    child_to_joint = {info['child']: (name, info) for name, info in joint_info.items()}
    cache = {}
    
    def build_path(child):
        if child in cache:
            return cache[child]
        if child not in child_to_joint:
            # Base link
            path = child
        else:
            joint_name, info = child_to_joint[child]
            parent_path = build_path(info['parent'])
            path = f"{parent_path}/{joint_name}/{child}"
        cache[child] = path
        return path
    
    joint_paths = {}
    for name, info in joint_info.items():
        joint_paths[name] = {
            'path': build_path(info['child']),
            'axis': info['axis']
        }
    
    return robot_name, joint_paths


def log_joint_angles(robot_name: str, joint_paths: Dict[str, Dict[str, any]], 
                     joint_names: list, joint_angles: list):
    """
    Log joint angles to Rerun for URDF visualization.
    
    Args:
        robot_name: Name of the robot from URDF
        joint_paths: Dictionary from build_joint_paths()
        joint_names: List of joint names in order
        joint_angles: List of joint angle values (radians) in same order as joint_names
    """
    for i, joint_name in enumerate(joint_names):
        if joint_name not in joint_paths:
            continue
        joint_angle = joint_angles[i]
        joint_data = joint_paths[joint_name]
        path = f"{robot_name}/{joint_data['path']}"
        axis = joint_data['axis']
        rotation = RotationAxisAngle(axis=axis, angle=joint_angle)
        rr.log(path, rr.Transform3D(rotation=rotation))


def init_rerun_with_urdf(app_id: str, urdf_path: Path, spawn: bool = True) -> Tuple[str, Dict[str, Dict[str, any]]]:
    """
    Initialize Rerun and load a URDF file.
    
    Args:
        app_id: Application ID for Rerun
        urdf_path: Path to the URDF file
        spawn: Whether to spawn the Rerun viewer
        
    Returns:
        Tuple of (robot_name, joint_paths_dict)
    """
    rr.init(app_id, spawn=spawn)
    rr.log_file_from_path(str(urdf_path))
    return build_joint_paths(urdf_path)
