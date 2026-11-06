#!/usr/bin/env python3
"""perception_node.py — object detection and 3D registration.

TODO:
  - Subscribe to /external_cam/image_raw and /external_cam/camera_info
  - Publish /object_position (5-10 Hz)
  - Segment the object in HSV
  - Find the contour center (u, v)
  - Project pixel -> 3D via K^{-1} and the camera->world TF
  - Intersect with the z=0 plane (table) to estimate the depth
  - Get cam_K / cam_T from the driver (params or CameraInfo)
"""

import cv2
import numpy as np
import rclpy
import rclpy.node
import xml.etree.ElementTree as ET
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image


HSV_MIN = np.array([160, 100, 100], dtype=np.uint8)
HSV_MAX = np.array([180, 255, 255], dtype=np.uint8)


class PerceptionNode(rclpy.node.Node):
    """Detects the object and publishes its 3D position."""

    def __init__(self):
        super().__init__("so101_perception")
        self._bridge = CvBridge()
        self._cam_k = None

        self.declare_parameter("cam_pos", [0.45, 0.25, 0.35])
        self.declare_parameter(
            "cam_xyaxes", [-0.707, 0.707, 0.0, -0.354, -0.354, 0.866]
        )

        self._cam_pos = np.array(self.get_parameter("cam_pos").value)
        xyaxes = np.array(self.get_parameter("cam_xyaxes").value)

        x_axis = xyaxes[0:3]
        y_axis = xyaxes[3:6]
        z_axis = np.cross(x_axis, y_axis)
        self._cam_rot = np.column_stack((x_axis, y_axis, z_axis))

        self._info_sub = self.create_subscription(
            CameraInfo, "/external_cam/camera_info", self._cb_camera_info, 10
        )
        self._image_sub = self.create_subscription(
            Image, "/external_cam/image_raw", self._cb_image, 10
        )
        self._pose_pub = self.create_publisher(PoseStamped, "/object_position", 10)
        self._annotated_pub = self.create_publisher(
            Image, "/perception/annotated_image", 10
        )
        self._dropbox_pub = self.create_publisher(PoseStamped, "/drop_box_position", 10)

        self.get_logger().info("Perception node ready.")

    def _cb_camera_info(self, msg):
        if self._cam_k is None:
            self._cam_k = np.array(msg.k).reshape(3, 3)

    def _cb_image(self, msg):
        if self._cam_k is None:
            return

        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        u, v, contour = self._detect_object(cv_image)

        if u is not None and v is not None:
            pos_3d = self._project_to_3d(u, v)
            if pos_3d is not None:
                self._publish_result(pos_3d, msg.header.stamp)

        # 1. Publish the static Drop Box Position
        box_msg = PoseStamped()
        box_msg.header.stamp = self.get_clock().now().to_msg()
        box_msg.header.frame_id = "world"
        box_msg.pose.position.x = 0.05
        box_msg.pose.position.y = -0.25
        box_msg.pose.position.z = 0.025
        box_msg.pose.orientation.w = 1.0
        self._dropbox_pub.publish(box_msg)

        # 2. Draw the contour of the cube
        if contour is not None:
            cv2.drawContours(cv_image, [contour], -1, (0, 255, 0), 1)

        # 3. Project the 3D Drop Box corners onto the 2D image
        # The box is centered at X=0.05, Y=-0.25 and is 0.10m wide.
        box_corners_3d = [
            (0.00, -0.20, 0.05),  # Top-Left
            (0.10, -0.20, 0.05),  # Top-Right
            (0.10, -0.30, 0.05),  # Bottom-Right
            (0.00, -0.30, 0.05),  # Bottom-Left
        ]

        box_pixels = []
        for pt in box_corners_3d:
            uv = self._project_3d_to_pixel(pt)
            if uv is not None:
                box_pixels.append(uv)

        if len(box_pixels) == 4:
            # Draw the projected polygon
            pts_array = np.array(box_pixels, dtype=np.int32)
            cv2.polylines(
                cv_image, [pts_array], isClosed=True, color=(255, 0, 0), thickness=1
            )

        # 3. Publish the annotated image to the NEW topic
        annotated_msg = self._bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        annotated_msg.header = msg.header
        self._annotated_pub.publish(annotated_msg)

    def _detect_object(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 50:
            return None, None

        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), largest_contour
        return None, None

    def _project_to_3d(self, u, v):
        """Projects a 2D pixel to the Z=0.025 plane using the known extrinsics."""
        ray_cv = np.linalg.inv(self._cam_k) @ np.array([u, v, 1.0])
        ray_mj = np.array([ray_cv[0], -ray_cv[1], -ray_cv[2]])
        ray_mj /= np.linalg.norm(ray_mj)
        ray_world = self._cam_rot @ ray_mj

        if abs(ray_world[2]) < 1e-6:
            return None

        t = (0.025 - self._cam_pos[2]) / ray_world[2]
        if t < 0:
            return None

        return self._cam_pos + t * ray_world

    def _project_3d_to_pixel(self, pt_3d):
        """Projects a 3D world coordinate back to a 2D camera pixel."""
        # Vector from camera to 3D point
        ray_world = np.array(pt_3d) - self._cam_pos
        # Reverse the rotation to get into MuJoCo camera frame
        ray_mj = self._cam_rot.T @ ray_world
        # Convert MuJoCo frame to OpenCV frame
        ray_cv = np.array([ray_mj[0], -ray_mj[1], -ray_mj[2]])

        if ray_cv[2] <= 0:
            return None  # Point is behind the camera

        # Normalize and apply intrinsics
        ray_cv /= ray_cv[2]
        uv = self._cam_k @ ray_cv
        return int(uv[0]), int(uv[1])

    def _publish_result(self, pos_3d, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.pose.position.x = pos_3d[0]
        msg.pose.position.y = pos_3d[1]
        msg.pose.position.z = pos_3d[2]
        self._pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
