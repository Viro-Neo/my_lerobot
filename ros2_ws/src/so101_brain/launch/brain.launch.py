#!/usr/bin/env python3
"""brain.launch.py — launches the brain of the arm (IK + pick & place)."""

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    env_use_sim = os.environ.get("USE_SIM", "false").lower()
    is_sim = env_use_sim == "true"

    return LaunchDescription(
        [
            Node(
                package="so101_brain",
                executable="brain",
                name="so101_brain",
                output="screen",
                parameters=[
                    {"use_sim": is_sim},
                ],
            ),
        ]
    )
