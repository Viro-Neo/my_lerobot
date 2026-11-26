import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Read the USE_SIM environment variable (defaults to false if not set)
    env_use_sim = os.environ.get("USE_SIM", "false").lower()
    env_phase = os.environ.get("PHASE", "1")
    is_sim = env_use_sim == "true"

    # Choose the correct extrinsic matrices based on USE_SIM
    if is_sim:
        cam_pos = [0.55, 0.35, 0.35] if env_phase == "2" else [0.45, 0.25, 0.35]
    else:
        cam_pos = [0.1295, -0.5699, 0.4229]

    cam_xyaxes = (
        [-0.707, 0.707, 0.0, -0.354, -0.354, 0.866]
        if is_sim
        else [0.9883, 0.1178, 0.0965, 0.1453, -0.5410, -0.8284]
    )

    return LaunchDescription(
        [
            # The Perception Node
            Node(
                package="so101_perception",
                executable="perception_node",  # Must match the console_scripts name in setup.py
                name="perception_node",
                output="screen",
                parameters=[
                    {"use_sim": is_sim},
                    {"cam_pos": cam_pos},
                    {"cam_xyaxes": cam_xyaxes},
                ],
            ),
            # Only launch the physical camera node if USE_SIM is NOT true
            Node(
                package="so101_perception",
                executable="camera_node",
                name="camera_node",
                output="screen",
                parameters=[
                    {"device_index": 2},
                    {"frame_id": "external_camera_optical_frame"},
                    {"fps": 30.0},
                    {"width": 640},
                    {"height": 480},
                    # {"fx": 800.36249},  # Logitech
                    # {"fy": 802.34293},
                    # {"cx": 329.38766},
                    # {"cy": 245.2717},
                    # {"k1": -0.016709},
                    # {"k2": 0.033700},
                    # {"p1": 0.005194},
                    # {"p2": 0.003060},
                    # {"k3": 0.000000},
                    {"fx": 509.41501},  # UGreen
                    {"fy": 507.61089},
                    {"cx": 310.92576},
                    {"cy": 201.87203},
                    {"k1": 0.000767},
                    {"k2": -0.014176},
                    {"p1": -0.000630},
                    {"p2": 0.004427},
                    {"k3": 0.000000},
                ],
                condition=UnlessCondition(env_use_sim),
            ),
            Node(
                package="so101_perception",
                executable="yolo_sort_perception_node"
                if env_phase == "2"
                else "yolo_perception_node",
                name="yolo_perception_node",
                output="screen",
                parameters=[
                    {"use_sim": is_sim},
                    {"cam_pos": cam_pos},
                    {"cam_xyaxes": cam_xyaxes},
                ],
            ),
        ]
    )
