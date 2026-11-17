import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Read the USE_SIM environment variable (defaults to false if not set)
    env_use_sim = os.environ.get("USE_SIM", "false").lower()
    is_sim = env_use_sim == "true"

    # Choose the correct extrinsic matrices based on USE_SIM
    cam_pos = [0.45, 0.25, 0.35] if is_sim else [0.0810, -0.5673, 0.4207]
    cam_xyaxes = (
        [-0.707, 0.707, 0.0, -0.354, -0.354, 0.866]
        if is_sim
        else [0.9939, 0.0615, 0.0913, 0.1101, -0.5456, -0.8308]
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
                condition=UnlessCondition(str(is_sim == "true")),
            ),
        ]
    )
