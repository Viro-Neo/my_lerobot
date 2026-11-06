"""
drop_box_marker_pub.py

Subscribes to /drop_box_position (PoseStamped) and publishes a
semi-transparent red Cube marker on /drop_box_marker for RViz.
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


class DropBoxMarkerPublisher(Node):
    """Publishes an RViz cube that follows the drop_box."""

    def __init__(self) -> None:
        super().__init__("drop_box_marker_pub")
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

    def _cb_box_position(self, msg: PoseStamped) -> None:
        self._marker.header.stamp = self.get_clock().now().to_msg()
        self._marker.pose = msg.pose
        self._pub.publish(self._marker)

    def _cb_obj_position(self, msg: PoseStamped) -> None:
        self._marker_obj.header.stamp = self.get_clock().now().to_msg()
        self._marker_obj.pose = msg.pose

        # Ensure the visual marker sits exactly on the table (Z = 0.025 for a 5cm cube)
        self._marker_obj.pose.position.z = 0.025
        self._pub_obj.publish(self._marker_obj)


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
