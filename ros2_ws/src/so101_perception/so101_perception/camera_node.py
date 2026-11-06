#!/usr/bin/env python3
"""camera_node.py - Captures from a USB camera and publishes to ROS 2 topics."""

import cv2
import rclpy
import rclpy.node
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo, Image


class CameraNode(rclpy.node.Node):
    """Publishes images and camera calibration info from a V4L2 USB camera."""

    def __init__(self):
        super().__init__("external_camera_publisher")

        self.declare_parameter("device_index", 2)
        self.declare_parameter("frame_id", "external_camera_optical_frame")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)

        self.declare_parameter("fx", 500.0)
        self.declare_parameter("fy", 500.0)
        self.declare_parameter("cx", 320.0)
        self.declare_parameter("cy", 240.0)

        self._device_index = self.get_parameter("device_index").value
        self._frame_id = self.get_parameter("frame_id").value
        self._fps = self.get_parameter("fps").value
        self._width = self.get_parameter("width").value
        self._height = self.get_parameter("height").value

        self._image_pub = self.create_publisher(Image, "/external_cam/image_raw", 1)
        self._info_pub = self.create_publisher(
            CameraInfo, "/external_cam/camera_info", 10
        )

        self._bridge = CvBridge()

        self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._device_index)

        if not self._cap.isOpened():
            self.get_logger().error(f"Cannot open camera device {self._device_index}")
            raise RuntimeError(f"Could not open /dev/video{self._device_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._camera_info_msg = self._build_camera_info()

        self._timer = self.create_timer(1.0 / self._fps, self._publish_frame)
        self.get_logger().info(
            f"Camera node running on device {self._device_index} at {self._fps} Hz ({self._width}x{self._height})."
        )

    def _build_camera_info(self) -> CameraInfo:
        """Constructs the intrinsic matrix K and projection matrix P."""
        fx = self.get_parameter("fx").value
        fy = self.get_parameter("fy").value
        cx = self.get_parameter("cx").value
        cy = self.get_parameter("cy").value

        info = CameraInfo()
        info.header.frame_id = self._frame_id
        info.width = self._width
        info.height = self._height
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        info.k = [
            fx,
            0.0,
            cx,
            0.0,
            fy,
            cy,
            0.0,
            0.0,
            1.0,
        ]

        info.r = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]

        info.p = [
            fx,
            0.0,
            cx,
            0.0,
            0.0,
            fy,
            cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return info

    def _publish_frame(self):
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self.get_logger().warning(
                "Failed to grab camera frame", throttle_duration_sec=2.0
            )
            return

        stamp = self.get_clock().now().to_msg()

        img_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self._frame_id

        self._camera_info_msg.header.stamp = stamp

        self._image_pub.publish(img_msg)
        self._info_pub.publish(self._camera_info_msg)

    def destroy_node(self):
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()
            self.get_logger().info("Camera device released.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
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
