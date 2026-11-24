#!/usr/bin/env python3
"""brain_phase2_node.py — Phase 2 color-sorting pick & place state machine."""

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


class SortState(Enum):
    IDLE = 0
    HOVER = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    MOVE_DROP = 5
    RELEASE = 6
    HOME = 7


class BrainPhase2Node(rclpy.node.Node):
    """Sorts red and blue cubes into their matching drop boxes."""

    def __init__(self):
        super().__init__("so101_brain_phase2")

        self.declare_parameter("use_sim", True)
        self._use_sim = self.get_parameter("use_sim").value
        self.grasp_z_offset = 0.015 if self._use_sim else -0.025

        self.model = None
        self.data = None
        self.q_current = np.zeros(len(JOINT_NAMES))
        self.ee_frame_id = None

        self.state = SortState.IDLE
        self.state_transition_time = time.time()

        self.is_running = False
        self.red_pos = None
        self.blue_pos = None
        self.red_done = False
        self.blue_done = False

        self.current_color = None
        self.current_target_pos = None

        # Drop positions matching scene_phase2.xml
        self.drop_pos_red = np.array([0.10, -0.20, 0.15])
        self.drop_pos_blue = np.array([0.10, 0.20, 0.15])
        self.home_pos = np.array([0.15, 0.0, 0.15])

        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, "/robot_description", self._cb_robot_description, qos
        )
        self.create_subscription(JointState, "/joint_states", self._cb_joint_states, 10)
        self.create_subscription(
            PoseStamped, "/phase2/red_cube_position", self._cb_red_pos, 10
        )
        self.create_subscription(
            PoseStamped, "/phase2/blue_cube_position", self._cb_blue_pos, 10
        )

        self.create_service(Trigger, "/brain/start_phase2", self._cb_start_phase2)
        self.create_service(Trigger, "/brain/stop_phase2", self._cb_stop_phase2)

        self._cmd_pub = self.create_publisher(JointState, "/joint_command", 10)
        self.create_timer(0.1, self._state_machine_step)
        self.get_logger().info(
            "Phase 2 Brain node ready. Trigger via /brain/start_phase2."
        )

    def _cb_robot_description(self, msg):
        if self.model is None:
            self.model = pin.buildModelFromXML(msg.data)
            self.data = self.model.createData()
            self.ee_frame_id = self.model.getFrameId("gripper_frame_link")

    def _cb_joint_states(self, msg):
        if self.model is None:
            return
        cmd_map = dict(zip(msg.name, msg.position))
        for i, name in enumerate(ARM_JOINT_NAMES):
            if name in cmd_map:
                self.q_current[i] = cmd_map[name]

    def _cb_red_pos(self, msg):
        if self.state == SortState.IDLE:
            self.red_pos = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            )

    def _cb_blue_pos(self, msg):
        if self.state == SortState.IDLE:
            self.blue_pos = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            )

    def _cb_start_phase2(self, request, response):
        self.is_running = True
        self.red_done = False
        self.blue_done = False
        response.success = True
        response.message = "Phase 2 color sorting started."
        self.get_logger().info("Received START_PHASE2 command.")
        return response

    def _cb_stop_phase2(self, request, response):
        self.is_running = False
        response.success = True
        response.message = "Phase 2 sorting stopped."
        return response

    def solve_ik(self, target_position, max_iter=500, eps=1e-4):
        if self.model is None:
            return None
        q_reference = np.array([0.0, -0.8, 1.2, 1.2, 0.0, 0.0])
        q = q_reference.copy()

        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            err = target_position - self.data.oMf[self.ee_frame_id].translation
            if np.linalg.norm(err) < eps:
                return q

            J = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]
            pinv_J = np.linalg.pinv(J)
            v = pinv_J @ err + (np.eye(self.model.nv) - pinv_J @ J) @ (
                (q_reference - q) * 0.5
            )
            q = np.clip(
                pin.integrate(self.model, q, v * 0.1),
                self.model.lowerPositionLimit,
                self.model.upperPositionLimit,
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
        self.state_transition_time = time.time()
        self.get_logger().info(f"[Phase2 - {self.current_color}] -> {self.state.name}")

    def _state_machine_step(self):
        if self.model is None:
            return
        elapsed = time.time() - self.state_transition_time

        if self.state == SortState.IDLE:
            if not self.is_running:
                return

            if not self.red_done and self.red_pos is not None:
                self.current_color = "red"
                self.current_target_pos = self.red_pos.copy()
                self._transition(SortState.HOVER)
            elif not self.blue_done and self.blue_pos is not None:
                self.current_color = "blue"
                self.current_target_pos = self.blue_pos.copy()
                self._transition(SortState.HOVER)
            elif self.red_done and self.blue_done:
                self.get_logger().info("Both cubes sorted! Returning to IDLE.")
                self.is_running = False
                self.current_color = None

        elif self.state == SortState.HOVER:
            target = self.current_target_pos.copy()
            target[2] += 0.15
            self.send_joint_command(self.solve_ik(target), gripper_pct=50.0)
            if elapsed > 5.0:
                self._transition(SortState.DESCEND)

        elif self.state == SortState.DESCEND:
            target = self.current_target_pos.copy()
            target[2] += self.grasp_z_offset
            self.send_joint_command(self.solve_ik(target), gripper_pct=50.0)
            if elapsed > 5.0:
                self._transition(SortState.GRASP)

        elif self.state == SortState.GRASP:
            target = self.current_target_pos.copy()
            target[2] += self.grasp_z_offset
            self.send_joint_command(self.solve_ik(target), gripper_pct=0.0)
            if elapsed > 5.0:
                self._transition(SortState.LIFT)

        elif self.state == SortState.LIFT:
            target = self.current_target_pos.copy()
            target[2] += 0.15
            self.send_joint_command(self.solve_ik(target), gripper_pct=0.0)
            if elapsed > 5.0:
                self._transition(SortState.MOVE_DROP)

        elif self.state == SortState.MOVE_DROP:
            drop_pos = (
                self.drop_pos_red if self.current_color == "red" else self.drop_pos_blue
            )
            self.send_joint_command(self.solve_ik(drop_pos), gripper_pct=0.0)
            if elapsed > 6.0:
                self._transition(SortState.RELEASE)

        elif self.state == SortState.RELEASE:
            release_target = (
                self.drop_pos_red.copy()
                if self.current_color == "red"
                else self.drop_pos_blue.copy()
            )
            release_target[0] -= 0.03
            release_target[1] += -0.03 if self.current_color == "red" else 0.03
            q_target = self.solve_ik(release_target)
            q_target[3] += 0.3
            self.send_joint_command(q_target, gripper_pct=50.0)
            if elapsed > 5.0:
                self._transition(SortState.HOME)

        elif self.state == SortState.HOME:
            self.send_joint_command(self.solve_ik(self.home_pos), gripper_pct=50.0)
            if elapsed > 5.0:
                if self.current_color == "red":
                    self.red_done = True
                    self.red_pos = None
                elif self.current_color == "blue":
                    self.blue_done = True
                    self.blue_pos = None
                self._transition(SortState.IDLE)


def main(args=None):
    rclpy.init(args=args)
    node = BrainPhase2Node()
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
