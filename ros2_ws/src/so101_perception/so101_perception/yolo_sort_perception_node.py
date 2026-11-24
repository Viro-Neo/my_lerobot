#!/usr/bin/env python3
"""yolo_sort_perception_node.py — Phase 2 multi-object (red & blue) YOLO-World detection."""

import time

import cv2
import numpy as np
import rclpy
import rclpy.node
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from ultralytics import YOLO


class YoloSortPerceptionNode(rclpy.node.Node):
    """Detects red and blue cubes separately and publishes their 3D positions."""

    def __init__(self):
        super().__init__("so101_yolo_sort_perception")
        self._bridge = CvBridge()
        self._cam_k = None
        self._cam_d = None

        self.model = YOLO("yolov8s-worldv2.pt")
        self.model.set_classes(["red cube", "blue cube", "cube", "block", "toy"])

        self.declare_parameter("use_sim", True)
        self.declare_parameter("cam_pos", [0.45, 0.25, 0.35])
        self.declare_parameter(
            "cam_xyaxes", [-0.707, 0.707, 0.0, -0.354, -0.354, 0.866]
        )

        self._use_sim = self.get_parameter("use_sim").value
        self._cam_pos = np.array(self.get_parameter("cam_pos").value)
        xyaxes = np.array(self.get_parameter("cam_xyaxes").value)

        x_axis = xyaxes[0:3]
        y_axis = xyaxes[3:6]
        z_axis = np.cross(x_axis, y_axis)
        self._cam_rot = np.column_stack((x_axis, y_axis, z_axis))

        self._last_infer_time = 0.0

        self._info_sub = self.create_subscription(
            CameraInfo, "/external_cam/camera_info", self._cb_camera_info, 10
        )
        self._image_sub = self.create_subscription(
            Image, "/external_cam/image_raw", self._cb_image, 1
        )

        # Separate 3D position topics for Phase 2
        self._red_pub = self.create_publisher(
            PoseStamped, "/phase2/red_cube_position", 10
        )
        self._blue_pub = self.create_publisher(
            PoseStamped, "/phase2/blue_cube_position", 10
        )
        self._annotated_pub = self.create_publisher(
            Image, "/perception/phase2_annotated_image", 10
        )

        self._tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_camera_tf()

        self.get_logger().info("Phase 2 YOLO Sort Perception node ready.")

    def _cb_camera_info(self, msg):
        if self._cam_k is None:
            self._cam_k = np.array(msg.k).reshape(3, 3)
            self._cam_d = np.array(msg.d)

    def _is_in_drop_box(self, pos_3d):
        """Returns True if the 3D point lies inside either the red or blue drop box."""
        x, y = pos_3d[0], pos_3d[1]
        in_red_box = (-0.02 <= x <= 0.12) and (-0.32 <= y <= -0.18)
        in_blue_box = (-0.02 <= x <= 0.12) and (0.18 <= y <= 0.32)
        return in_red_box or in_blue_box

    def _cb_image(self, msg):
        if self._cam_k is None or self._cam_d is None:
            return

        # Throttle to 5 Hz to keep RViz2 responsive
        now = time.time()
        if (now - self._last_infer_time) < 0.2:
            return
        self._last_infer_time = now

        raw_cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if not self._use_sim:
            cv_image = cv2.undistort(raw_cv_image, self._cam_k, self._cam_d)
        else:
            cv_image = raw_cv_image

        detections = self._detect_objects(cv_image)

        for color, (u, v, contour, label) in detections.items():
            pos_3d = self._project_to_3d(u, v)
            if pos_3d is None or self._is_in_drop_box(pos_3d):
                continue

            if color == "red":
                self._publish_pose(self._red_pub, pos_3d, msg.header.stamp)
                draw_color = (0, 0, 255)
            else:
                self._publish_pose(self._blue_pub, pos_3d, msg.header.stamp)
                draw_color = (255, 0, 0)

            cv2.drawContours(cv_image, [contour], -1, draw_color, 2)
            x1, y1 = contour[0][0], contour[0][1]
            cv2.putText(
                cv_image,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                draw_color,
                1,
            )

        # Draw both static drop boxes on the annotated image
        self._draw_box(cv_image, 0.05, -0.25, (0, 0, 255))  # Red drop box
        self._draw_box(cv_image, 0.05, 0.25, (255, 0, 0))  # Blue drop box

        annotated_msg = self._bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        annotated_msg.header = msg.header
        self._annotated_pub.publish(annotated_msg)

    def _detect_objects(self, cv_image):
        """Returns a dict mapping 'red' and/or 'blue' to (u, v, contour, label)."""
        results = self.model(cv_image, conf=0.02, imgsz=640, verbose=False)
        found = {}
        image_area = cv_image.shape[0] * cv_image.shape[1]

        for result in results:
            for box_data in result.boxes:
                box = box_data.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                box_area = (x2 - x1) * (y2 - y1)
                if box_area >= (image_area * 0.15) or box_area < 50:
                    continue

                crop = cv_image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                if crop.size == 0:
                    continue

                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                crop_pixels = crop.shape[0] * crop.shape[1]

                # Red ratio
                r1 = cv2.inRange(
                    hsv_crop, np.array([0, 70, 50]), np.array([15, 255, 255])
                )
                r2 = cv2.inRange(
                    hsv_crop, np.array([160, 70, 50]), np.array([179, 255, 255])
                )
                red_ratio = np.count_nonzero(cv2.bitwise_or(r1, r2)) / crop_pixels

                # Blue ratio
                b_mask = cv2.inRange(
                    hsv_crop, np.array([95, 70, 50]), np.array([135, 255, 255])
                )
                blue_ratio = np.count_nonzero(b_mask) / crop_pixels

                conf = float(box_data.conf[0])
                u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
                contour = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
                )

                if red_ratio >= 0.15 and red_ratio > blue_ratio and "red" not in found:
                    found["red"] = (u, v, contour, f"red_cube {conf:.2f}")
                elif (
                    blue_ratio >= 0.15
                    and blue_ratio > red_ratio
                    and "blue" not in found
                ):
                    found["blue"] = (u, v, contour, f"blue_cube {conf:.2f}")

                if len(found) == 2:
                    return found

        return found

    def _draw_box(self, cv_image, cx, cy, color):
        corners = [
            (cx - 0.05, cy + 0.05, 0.05),
            (cx + 0.05, cy + 0.05, 0.05),
            (cx + 0.05, cy - 0.05, 0.05),
            (cx - 0.05, cy - 0.05, 0.05),
        ]
        pixels = [self._project_3d_to_pixel(pt) for pt in corners]
        if all(p is not None for p in pixels):
            cv2.polylines(cv_image, [np.array(pixels, dtype=np.int32)], True, color, 1)

    def _project_to_3d(self, u, v):
        ray_cv = np.linalg.inv(self._cam_k) @ np.array([u, v, 1.0])
        if self._use_sim:
            ray_mj = np.array([ray_cv[0], -ray_cv[1], -ray_cv[2]])
            ray_mj /= np.linalg.norm(ray_mj)
            ray_world = self._cam_rot @ ray_mj
        else:
            ray_cv /= np.linalg.norm(ray_cv)
            ray_world = self._cam_rot @ ray_cv

        if abs(ray_world[2]) < 1e-6:
            return None
        t = (0.025 - self._cam_pos[2]) / ray_world[2]
        if t < 0:
            return None
        return self._cam_pos + t * ray_world

    def _project_3d_to_pixel(self, pt_3d):
        ray_world = np.array(pt_3d) - self._cam_pos
        if self._use_sim:
            ray_mj = self._cam_rot.T @ ray_world
            ray_cv = np.array([ray_mj[0], -ray_mj[1], -ray_mj[2]])
        else:
            ray_cv = self._cam_rot.T @ ray_world

        if ray_cv[2] <= 0:
            return None
        ray_cv /= ray_cv[2]
        uv = self._cam_k @ ray_cv
        return int(uv[0]), int(uv[1])

    def _publish_pose(self, pub, pos_3d, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(pos_3d[0])
        msg.pose.position.y = float(pos_3d[1])
        msg.pose.position.z = float(pos_3d[2])
        pub.publish(msg)

    def _publish_camera_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "external_camera_optical_frame"
        t.transform.translation.x = float(self._cam_pos[0])
        t.transform.translation.y = float(self._cam_pos[1])
        t.transform.translation.z = float(self._cam_pos[2])

        R = self._cam_rot
        trace = np.trace(R)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w, x, y, z = (
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            )
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w, x, y, z = (
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            )
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w, x, y, z = (
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            )
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w, x, y, z = (
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
            )

        t.transform.rotation.x = float(x)
        t.transform.rotation.y = float(y)
        t.transform.rotation.z = float(z)
        t.transform.rotation.w = float(w)
        self._tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = YoloSortPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
