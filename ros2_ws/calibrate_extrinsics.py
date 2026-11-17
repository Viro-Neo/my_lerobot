#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class ExtrinsicCalibrator(Node):
    def __init__(self):
        super().__init__("extrinsic_calibrator")
        self.bridge = CvBridge()
        self.cam_k = None
        self.cam_d = None

        self.create_subscription(
            CameraInfo, "/external_cam/camera_info", self.info_cb, 10
        )
        self.create_subscription(Image, "/external_cam/image_raw", self.img_cb, 10)

        # Updated ArUco syntax for OpenCV 4.10+
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # 100mm marker 3D coordinates
        marker_size = 0.10
        self.obj_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )

        self.get_logger().info("Waiting for camera feed...")

    def info_cb(self, msg):
        if self.cam_k is None:
            self.cam_k = np.array(msg.k).reshape(3, 3)
            self.cam_d = np.array(msg.d)

    def img_cb(self, msg):
        if self.cam_k is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Updated detection call
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            ret, rvec, tvec = cv2.solvePnP(
                self.obj_points, corners[0], self.cam_k, self.cam_d
            )

            if ret:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, self.cam_k, self.cam_d, rvec, tvec, 0.05)

                # Math: Extract Camera pose in Marker frame
                R_cm, _ = cv2.Rodrigues(rvec)
                R_mc = R_cm.T
                t_mc = -R_mc @ tvec

                # Add the 20cm base_link offset to the marker's X position
                cam_pos = t_mc + np.array([[0.20], [0.0], [0.0]])

                # Extract the Camera's X and Y rotation axes
                x_axis = R_mc[:, 0]
                y_axis = R_mc[:, 1]

                self.get_logger().info(
                    f"cam_pos: [{cam_pos[0][0]:.4f}, {cam_pos[1][0]:.4f}, {cam_pos[2][0]:.4f}]"
                )
                self.get_logger().info(
                    f"cam_xyaxes: [{x_axis[0]:.4f}, {x_axis[1]:.4f}, {x_axis[2]:.4f}, {y_axis[0]:.4f}, {y_axis[1]:.4f}, {y_axis[2]:.4f}]"
                )

        cv2.imshow("Extrinsics - Align Red Forward, Green Left", frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ExtrinsicCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
