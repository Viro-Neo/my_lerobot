"""
drop_box_marker_pub.py

Phase 1: Publishes /drop_box_marker and /object_marker from /drop_box_position and /object_position.
Phase 2: Publishes a MarkerArray on /phase2/markers with both colored drop boxes and both cubes.
"""

import os

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class DropBoxMarkerPublisher(Node):
    """Publishes an RViz cube that follows the drop_box."""

    def __init__(self) -> None:
        super().__init__("drop_box_marker_pub")
        self._phase = os.environ.get("PHASE", "1")

        # --- PHASE 1 MARKERS (unchanged) ---
        self._sub = self.create_subscription(
            PoseStamped, "drop_box_position", self._cb_box_position, 10
        )
        self._pub = self.create_publisher(Marker, "drop_box_marker", 10)
        # 10cm × 10cm × 5cm cube (same size as the MuJoCo drop_box)
        self._marker = Marker()
        self._marker.header.frame_id = "world"
        self._marker.ns = "drop_box"
        self._marker.id = 0
        self._marker.type = Marker.CUBE
        self._marker.action = Marker.ADD
        self._marker.scale.x = 0.10
        self._marker.scale.y = 0.10
        self._marker.scale.z = 0.05
        self._marker.color.r = 0.55
        self._marker.color.g = 0.27
        self._marker.color.b = 0.07
        self._marker.color.a = 0.6  # semi-transparent
        self._marker.lifetime.sec = 0

        # --- OBJECT CUBE MARKER ---
        self._sub_obj = self.create_subscription(
            PoseStamped, "object_position", self._cb_obj_position, 10
        )
        self._pub_obj = self.create_publisher(Marker, "object_marker", 10)

        self._marker_obj = Marker()
        self._marker_obj.header.frame_id = "world"
        self._marker_obj.ns = "object"
        self._marker_obj.id = 1
        self._marker_obj.type = Marker.CUBE
        self._marker_obj.action = Marker.ADD
        self._marker_obj.scale.x = 0.05  # 5cm cube
        self._marker_obj.scale.y = 0.05
        self._marker_obj.scale.z = 0.05
        self._marker_obj.color.r = 1.0
        self._marker_obj.color.g = 0.0
        self._marker_obj.color.b = 0.0
        self._marker_obj.color.a = 1.0  # Solid color

        # --- PHASE 2 MARKERS (Red & Blue boxes + Red & Blue cubes) ---
        if self._phase == "2":
            self._p2_pub = self.create_publisher(MarkerArray, "/phase2/markers", 10)
            self.create_subscription(
                PoseStamped, "/phase2/red_cube_position", self._cb_p2_red, 10
            )
            self.create_subscription(
                PoseStamped, "/phase2/blue_cube_position", self._cb_p2_blue, 10
            )
            # Publish static drop boxes at 2 Hz so late-joining RViz sessions always see them
            self.create_timer(0.5, self._publish_p2_boxes)

    def _cb_box_position(self, msg: PoseStamped) -> None:
        self._marker.header.stamp = self.get_clock().now().to_msg()
        self._marker.header.frame_id = "base_link"
        self._marker.pose = msg.pose
        self._pub.publish(self._marker)

    def _cb_obj_position(self, msg: PoseStamped) -> None:
        self._marker_obj.header.stamp = self.get_clock().now().to_msg()
        self._marker_obj.header.frame_id = "base_link"
        self._marker_obj.pose = msg.pose
        # self._marker_obj.pose.position.z = 0.025
        self._pub_obj.publish(self._marker_obj)

    def _make_cube_marker(
        self,
        marker_id: int,
        ns: str,
        scale: tuple,
        rgba: tuple,
        xyz: tuple,
        lifetime_sec: int,
    ) -> Marker:
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.ns = ns
        m.id = marker_id
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(xyz[0])
        m.pose.position.y = float(xyz[1])
        m.pose.position.z = float(xyz[2])
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.lifetime = Duration(sec=lifetime_sec)
        return m

    def _publish_p2_boxes(self) -> None:
        arr = MarkerArray()
        # Positions match scene_phase2.xml: red at (0.05, -0.25), blue at (0.05, 0.25)
        arr.markers.append(
            self._make_cube_marker(
                0,
                "p2_boxes",
                (0.10, 0.10, 0.05),
                (0.85, 0.10, 0.15, 0.5),
                (0.05, -0.25, 0.025),
                0,
            )
        )
        arr.markers.append(
            self._make_cube_marker(
                1,
                "p2_boxes",
                (0.10, 0.10, 0.05),
                (0.10, 0.30, 0.85, 0.5),
                (0.05, 0.25, 0.025),
                0,
            )
        )
        self._p2_pub.publish(arr)

    def _cb_p2_red(self, msg: PoseStamped) -> None:
        arr = MarkerArray()
        arr.markers.append(
            self._make_cube_marker(
                2,
                "p2_cubes",
                (0.05, 0.05, 0.05),
                (1.0, 0.0, 0.0, 1.0),
                (msg.pose.position.x, msg.pose.position.y, 0.025),
                1,
            )
        )
        self._p2_pub.publish(arr)

    def _cb_p2_blue(self, msg: PoseStamped) -> None:
        arr = MarkerArray()
        arr.markers.append(
            self._make_cube_marker(
                3,
                "p2_cubes",
                (0.05, 0.05, 0.05),
                (0.1, 0.4, 1.0, 1.0),
                (msg.pose.position.x, msg.pose.position.y, 0.025),
                1,
            )
        )
        self._p2_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = DropBoxMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
