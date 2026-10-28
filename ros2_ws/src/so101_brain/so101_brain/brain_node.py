#!/usr/bin/env python3
"""brain_node.py - intelligence of the arm: kinematics, then pick & place.

Never talks to the backend: reads /joint_states, commands through /joint_command.

TODO:
  - Forward kinematics: from /joint_states, publish the TCP pose
    (e.g. /end_effector_pose, geometry_msgs/PoseStamped) and show it in RViz
  - Inverse kinematics: reach an XYZ target (ikpy and pinocchio are installed)
  - Later: pick & place state machine
"""

# import numpy as np
# import pinocchio as pin
# import rclpy
# import rclpy.node
# from rclpy.qos import QoSProfile, QoSDurabilityPolicy
# from rclpy.executors import ExternalShutdownException
# from geometry_msgs.msg import PoseStamped
# from sensor_msgs.msg import JointState
# from std_msgs.msg import String

import time
import numpy as np
import pinocchio as pin
import rclpy
import rclpy.node
from enum import Enum
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]


class BrainNode(rclpy.node.Node):
    """Brain of the arm: kinematics, then pick & place."""

    def __init__(self):
        super().__init__("so101_brain")

        self.model = None
        self.data = None
        self.q_current = np.zeros(len(JOINT_NAMES))
        self.ee_frame_id = None

        # Subscribe to robot_description (Transient Local) to build Pinocchio model
        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, "/robot_description", self._cb_robot_description, qos
        )

        self.create_subscription(JointState, "/joint_states", self._cb_joint_states, 10)
        self._tcp_pub = self.create_publisher(PoseStamped, "/end_effector_pose", 10)
        self._cmd_pub = self.create_publisher(JointState, "/joint_command", 10)

        self.get_logger().info("Brain node ready.")

    def _cb_robot_description(self, msg):
        if self.model is None:
            self.model = pin.buildModelFromXML(msg.data)
            self.data = self.model.createData()
            self.ee_frame_id = self.model.getFrameId("gripper_link")
            self.get_logger().info("Pinocchio model built successfully.")

    def _cb_joint_states(self, msg):
        if self.model is None:
            return

        # Extract ALL joint positions (arm + gripper) in the correct order
        cmd_map = dict(zip(msg.name, msg.position))
        for i, name in enumerate(JOINT_NAMES):
            if name in cmd_map:
                self.q_current[i] = cmd_map[name]

        self._forward_kinematics(self.q_current)

    def _forward_kinematics(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        tcp_placement = self.data.oMf[self.ee_frame_id]
        self._publish_end_effector_pose(tcp_placement)

    def _publish_end_effector_pose(self, placement):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.pose.position.x = placement.translation[0]
        msg.pose.position.y = placement.translation[1]
        msg.pose.position.z = placement.translation[2]

        quat = pin.Quaternion(placement.rotation)
        msg.pose.orientation.x = quat.x
        msg.pose.orientation.y = quat.y
        msg.pose.orientation.z = quat.z
        msg.pose.orientation.w = quat.w

        self._tcp_pub.publish(msg)

    def solve_ik(self, target_position, target_rotation=None, max_iter=500, eps=1e-4):
        """Solves IK for a desired XYZ position (and optional rotation) using pseudo-inverse."""
        if self.model is None:
            return None

        q = self.q_current.copy()

        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            placement = self.data.oMf[self.ee_frame_id]
            err = target_position - placement.translation

            if np.linalg.norm(err) < eps:
                return q

            J = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            J_pos = J[:3, :]  # Use only position Jacobian

            v = np.linalg.pinv(J_pos) @ err
            q = pin.integrate(self.model, q, v * 0.1)  # 0.1 step size

        self.get_logger().warning("IK failed to converge.")
        return q

    def send_joint_command(self, q, gripper_pct=100.0):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(q) + [gripper_pct]
        self._cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
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
