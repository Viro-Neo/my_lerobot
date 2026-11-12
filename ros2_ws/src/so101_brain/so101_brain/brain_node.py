#!/usr/bin/env python3
"""brain_node.py - intelligence of the arm: kinematics, then pick & place.

Never talks to the backend: reads /joint_states, commands through /joint_command.

TODO:
  - Forward kinematics: from /joint_states, publish the TCP pose
    (e.g. /end_effector_pose, geometry_msgs/PoseStamped) and show it in RViz
  - Inverse kinematics: reach an XYZ target (ikpy and pinocchio are installed)
  - Later: pick & place state machine
"""

import time
from enum import Enum

import numpy as np
import pinocchio as pin
import rclpy
import rclpy.node
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]


class PickPlaceState(Enum):
    IDLE = 0
    HOVER = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    MOVE_DROP = 5
    RELEASE = 6
    HOME = 7


class BrainNode(rclpy.node.Node):
    """Brain of the arm: kinematics, then pick & place."""

    def __init__(self):
        super().__init__("so101_brain")

        self.model = None
        self.data = None
        self.q_current = np.zeros(len(JOINT_NAMES))
        self.ee_frame_id = None

        self.state = PickPlaceState.IDLE
        self.state_start_time = time.time()
        self.target_obj_pos = None

        self.is_running = False

        self.drop_pos = np.array([0.10, -0.20, 0.15])
        self.home_pos = np.array([0.15, 0.0, 0.15])

        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, "/robot_description", self._cb_robot_description, qos
        )
        self.create_subscription(JointState, "/joint_states", self._cb_joint_states, 10)
        self.create_subscription(
            PoseStamped, "/object_position", self._cb_object_position, 10
        )

        self.create_service(Trigger, "/brain/start", self._cb_start)
        self.create_service(Trigger, "/brain/stop", self._cb_stop)

        self._tcp_pub = self.create_publisher(PoseStamped, "/end_effector_pose", 10)
        self._cmd_pub = self.create_publisher(JointState, "/joint_command", 10)

        self.create_timer(0.1, self._state_machine_step)
        self.get_logger().info("Brain node ready.")

    def _cb_robot_description(self, msg):
        if self.model is None:
            self.model = pin.buildModelFromXML(msg.data)
            self.data = self.model.createData()
            self.ee_frame_id = self.model.getFrameId("gripper_frame_link")
            self.get_logger().info("Pinocchio model built successfully.")

    def _cb_joint_states(self, msg):
        if self.model is None:
            return
        cmd_map = dict(zip(msg.name, msg.position))
        for i, name in enumerate(ARM_JOINT_NAMES):
            if name in cmd_map:
                self.q_current[i] = cmd_map[name]
        self._forward_kinematics(self.q_current)

    def _cb_object_position(self, msg):
        if self.state == PickPlaceState.IDLE:
            self.target_obj_pos = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            )

    def _cb_start(self, request, response):
        self.is_running = True
        response.success = True
        response.message = "Brain started. Will pick up object when detected."
        self.get_logger().info("Received START command.")
        return response

    def _cb_stop(self, request, response):
        self.is_running = False
        response.success = True
        response.message = (
            "Brain stopped. Will return to IDLE after current cycle or wait."
        )
        self.get_logger().info("Received STOP command.")
        return response

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
        if self.model is None:
            return None

        q_reference = np.array([0.0, -0.8, 1.2, 1.2, 0.0, 0.0])
        q = q_reference.copy()

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
            J_pos = J[:3, :]

            v = np.linalg.pinv(J_pos) @ err

            null_space_matrix = np.eye(self.model.nv) - np.linalg.pinv(J_pos) @ J_pos
            posture_err = q_reference - q

            v += null_space_matrix @ (posture_err * 0.5)

            q = pin.integrate(self.model, q, v * 0.1)

            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)

        self.get_logger().warning(
            "IK failed to converge. Attempting to use best effort."
        )
        return q

    def send_joint_command(self, q, gripper_pct=100.0):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(q[:5]) + [float(gripper_pct)]
        self._cmd_pub.publish(msg)

    def _transition(self, next_state):
        self.state = next_state
        self.state_start_time = time.time()
        self.get_logger().info(f"Transitioning to: {self.state.name}")

    def _state_machine_step(self):
        if self.model is None:
            return
        elapsed = time.time() - self.state_start_time

        if self.state == PickPlaceState.IDLE:
            if self.is_running and self.target_obj_pos is not None:
                self._transition(PickPlaceState.HOVER)

        elif self.state == PickPlaceState.HOVER:
            target = self.target_obj_pos.copy()
            target[2] += 0.15
            q_target = self.solve_ik(target)
            self.send_joint_command(q_target, gripper_pct=50.0)
            if elapsed > 5.0:
                self._transition(PickPlaceState.DESCEND)

        elif self.state == PickPlaceState.DESCEND:
            target = self.target_obj_pos.copy()
            target[2] += 0.015
            q_target = self.solve_ik(target)
            self.send_joint_command(q_target, gripper_pct=50.0)
            if elapsed > 5.0:
                self._transition(PickPlaceState.GRASP)

        elif self.state == PickPlaceState.GRASP:
            target = self.target_obj_pos.copy()
            target[2] += 0.015
            q_target = self.solve_ik(target)
            self.send_joint_command(q_target, gripper_pct=0.0)
            if elapsed > 5.0:
                self._transition(PickPlaceState.LIFT)

        elif self.state == PickPlaceState.LIFT:
            target = self.target_obj_pos.copy()
            target[2] += 0.15
            q_target = self.solve_ik(target)
            self.send_joint_command(q_target, gripper_pct=0.0)
            if elapsed > 5.0:
                self._transition(PickPlaceState.MOVE_DROP)

        elif self.state == PickPlaceState.MOVE_DROP:
            q_target = self.solve_ik(self.drop_pos)
            self.send_joint_command(q_target, gripper_pct=0.0)
            if elapsed > 5.0:
                self._transition(PickPlaceState.RELEASE)

        elif self.state == PickPlaceState.RELEASE:
            release_target = self.drop_pos.copy()
            release_target[0] -= 0.03
            release_target[1] -= 0.03

            q_target = self.solve_ik(release_target)

            if q_target is not None:
                q_target[3] += 0.25

            self.send_joint_command(q_target, gripper_pct=50.0)
            if elapsed > 5.0:
                self._transition(PickPlaceState.HOME)

        elif self.state == PickPlaceState.HOME:
            q_target = self.solve_ik(self.home_pos)
            self.send_joint_command(q_target, gripper_pct=50.0)
            if elapsed > 5.0:
                pass


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
