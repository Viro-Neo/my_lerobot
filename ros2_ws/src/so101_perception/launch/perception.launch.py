import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import UnlessCondition
from launch_ros.actions import Node


def generate_launch_description():
    # Read the USE_SIM environment variable (defaults to false if not set)
    env_use_sim = os.environ.get("USE_SIM", "false").lower()

    return LaunchDescription(
        [
            # The Perception Node
            Node(
                package="so101_perception",
                executable="perception_node",  # Must match the console_scripts name in setup.py
                name="perception_node",
                output="screen",
                parameters=[
                    {
                        "cam_pos": [0.45, 0.25, 0.35]
                    },  # Example of overriding the default
                    {"cam_xyaxes": [-0.707, 0.707, 0, -0.354, -0.354, 0.866]},
                ],
            ),
            # Only launch the physical camera node if USE_SIM is NOT true
            Node(
                package="so101_perception",
                executable="camera_node",
                name="camera_node",
                output="screen",
                condition=UnlessCondition(str(env_use_sim == "true")),
            ),
        ]
    )
