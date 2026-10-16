#!/usr/bin/env python3
"""driver_node.py - ROS2 driver of the SO-ARM101.

The only node that talks to the backend: SO101Sim (MuJoCo) or SO101Follower
(real arm), chosen by the `use_sim` parameter. The rest of the stack never
knows which one runs.

TODO:
  - Publish /joint_states in RADIANS (>= 20 Hz)
  - Subscribe to /joint_command (radians, gripper 0-100 %)
"""

from pathlib import Path

import rclpy
import rclpy.node
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from so101_sim import SO101Sim

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Gripper joint limits in the URDF (rad).
# LeRobot convention: 0 % = closed (lower), 100 % = open (upper).
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)

# Calibration file of the real arm, mounted by docker compose (CALIBRATION_FILE in docker/.env).
CALIBRATION_FILE = Path("/calibration/arm.json")
DEFAULT_PORT = "/dev/ttyACM0"

CONTROL_HZ = 50.0
JOINT_STATE_HZ = 30.0


def _gripper_pct_to_rad(pct: float) -> float:
    pct = max(0.0, min(100.0, pct))  # clamp between 0 and 100
    low, high = GRIPPER_RANGE_RAD
    return low + (pct / 100.0) * (high - low)


class DriverNode(rclpy.node.Node):
    """Driver SO-ARM101: /joint_command -> backend -> /joint_states."""

    def __init__(self):
        super().__init__("so101_driver")
        self._use_sim: bool = self.declare_parameter("use_sim", True).value
        port: str = self.declare_parameter("port", DEFAULT_PORT).value
        mode = "simulation" if self._use_sim else "hardware"
        self.get_logger().info(f"Mode: {mode} (port={port})")

        self._robot = self._connect_arm(port)
        self.get_logger().info(f"Robot connected: {type(self._robot).__name__}")
        self.get_logger().info(f"Obs: {self._robot.get_observation()}")

        self.get_logger().info(f"Trad: {list(self._robot.get_observation().values())}")

        self._joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._joint_sub = self.create_subscription(
            JointState, "/joint_command", self._cb_joint_command, 10
        )
        self._state_timer = self.create_timer(
            1.0 / JOINT_STATE_HZ, self._publish_joint_states
        )

        self._current_cmd = None
        self._target_cmd = None
        self._arm_max_speed = 1.0  # radians per second
        self._gripper_max_speed = 50.0  # percent per second

        self._control_timer = self.create_timer(1.0 / CONTROL_HZ, self._control_step)
        self.get_logger().info("Driver node ready.")

    def _connect_arm(self, port: str = DEFAULT_PORT) -> SO101Sim | SO101Follower:
        """Creates and connects the backend chosen by use_sim (provided)."""
        if self._use_sim:
            robot = SO101Sim()
            robot.connect()
            return robot

        if not CALIBRATION_FILE.is_file():
            raise RuntimeError(
                f"No calibration file at {CALIBRATION_FILE}: set CALIBRATION_FILE in docker/.env"
            )
        config = SO101FollowerConfig(
            port=port,
            id=CALIBRATION_FILE.stem,
            calibration_dir=CALIBRATION_FILE.parent,
            use_radians=True,
        )
        robot = SO101Follower(config)
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            robot.disconnect()
            raise RuntimeError(
                f"{CALIBRATION_FILE} does not match the motors: wrong file for this arm? "
                "Re-run lerobot-calibrate on the host."
            )
        # robot.bus.disable_torque()  # Comment out this line to control the robot.
        self.get_logger().info(
            "Torque disabled: arm can be moved freely by hand. Remove this part to control the arm."
        )
        return robot

    def destroy_node(self):
        if self._robot.is_connected:
            self._robot.disconnect()
            self.get_logger().info("Robot disconnected.")
        super().destroy_node()

    def _cb_joint_command(self, msg: JointState):
        # Map incoming joint names to their positions
        cmd_map = dict(zip(msg.name, msg.position))

        # On the very first command, seed current_cmd with current positions
        if self._current_cmd is None:
            obs = self._robot.get_observation()
            self._current_cmd = {}
            for name in JOINT_NAMES:
                key = f"{name}.pos" if f"{name}.pos" in obs else name
                self._current_cmd[name] = float(obs[key])

        if self._target_cmd is None:
            self._target_cmd = dict(self._current_cmd)

        for name in JOINT_NAMES:
            if name in cmd_map:
                self._target_cmd[name] = float(cmd_map[name])

     def _control_step(self):
        if self._current_cmd is None or self._target_cmd is None:
            return

        dt = 1.0 / CONTROL_HZ
        action = {}
        moved = False

        for name in JOINT_NAMES:
            target = self._target_cmd[name]
            current = self._current_cmd[name]
            diff = target - current

            # Use percent speed limit for gripper, rad/s for arm joints
            speed = (
                self._gripper_max_speed if name == "gripper" else self._arm_max_speed
            )
            max_step = speed * dt

            if abs(diff) <= max_step:
                self._current_cmd[name] = target
            else:
                self._current_cmd[name] += max_step if diff > 0 else -max_step
                moved = True

            key = (
                f"{name}.pos"
                if f"{name}.pos" in self._robot.get_observation()
                else name
            )
            action[key] = self._current_cmd[name]

        # Send action smoothly
        self._robot.send_action(action)

    def _publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        pos_trad = list(self._robot.get_observation().values())
        pos_trad[5] = _gripper_pct_to_rad(pos_trad[5])
        msg.position = pos_trad
        self._joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
