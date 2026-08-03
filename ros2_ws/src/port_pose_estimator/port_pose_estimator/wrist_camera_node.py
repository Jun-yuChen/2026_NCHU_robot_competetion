#!/usr/bin/env python3
"""
rs_camera_node

Wraps RsCamera (RealSense D405) as a ROS 2 publisher node. Each cycle it
reads one synchronized RGB-D frame, packs color image + depth image +
camera intrinsics into a single perception_msgs/RGBDFrame message, and
publishes it.

Parameters:
    serial    (string, required)  RealSense device serial number.
    width     (int, default 848)
    height    (int, default 480)
    fps       (int, default 30)
    frame_id  (string, default 'camera_color_optical_frame')
    topic     (string, default 'camera/rgbd')
"""

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo

from custom_interface.msg import RealsenseFrame
from .utils.camera import RsCamera, get_connected_serials


class WristCameraNode(Node):
    def __init__(self):
        super().__init__('wrist_camera_node')

        self.declare_parameter('serial', '')
        self.declare_parameter('width', 848)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('frame_id', 'wrist_camera_frame')
        self.declare_parameter('topic', 'camera/rgbd')

        serial = self.get_parameter('serial').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
        self.frame_id = self.get_parameter('frame_id').value
        topic = self.get_parameter('topic').value

        if not serial:
            available = get_connected_serials()
            self.get_logger().error(
                f"Parameter 'serial' is required. Connected devices: {available}"
            )
            raise RuntimeError("Missing required parameter 'serial'.")

        self.bridge = CvBridge()

        # RsCamera.__init__ blocks for ~30 warm-up frames (up to 5s each on
        # failure) before returning, which is fine here since it runs once
        # before rclpy.spin() starts — it's not inside a callback.
        self.camera = RsCamera(serial=serial, width=width, height=height, fps=fps, name='wrist_camera')

        self.pub = self.create_publisher(RealsenseFrame, topic, 10)

        # A wall timer at the stream's own fps is simpler than a spin_once
        # polling loop here: unlike ArmCmd, this node has no other
        # subscriptions it needs to keep serviced between reads, so a plain
        # timer callback is sufficient and keeps the executor in charge of
        # timing.
        period = 1.0 / float(fps)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(f"Publishing RealsenseFrame on '{topic}' at ~{fps} Hz (serial={serial}).")

    def build_camera_info(self, intr: dict, header: Header) -> CameraInfo:
        info = CameraInfo()
        info.header = header
        info.width = intr['width']
        info.height = intr['height']
        info.distortion_model = intr['model']
        info.d = [float(c) for c in intr['coeffs']]

        fx, fy = float(intr['fx']), float(intr['fy'])
        cx, cy = float(intr['ppx']), float(intr['ppy'])

        info.k = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        return info

    def timer_callback(self):
        # NOTE: RsCamera.read() calls pipeline.wait_for_frames(5000), which
        # blocks the calling thread. In the normal case a frame is ready
        # well within one timer period, so this doesn't stall the executor.
        # It only becomes a problem if frames stop arriving entirely (e.g.
        # cable pulled) — the executor will stall for up to 5s in that case.
        # If that's a concern, move the read loop to its own thread.
        frame = self.camera.read()
        if frame is None:
            self.get_logger().warn('Dropped/incomplete frameset, skipping publish.')
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        rgb_msg = self.bridge.cv2_to_imgmsg(frame.color_image, encoding='bgr8')
        rgb_msg.header = header

        # RealSense z16 depth is raw uint16 depth units. Publish it as-is
        # in 16UC1, same convention realsense-ros uses for
        # /depth/image_rect_raw. The scale factor to convert to meters
        # (meters = raw_value * depth_scale) is carried in msg.depth_scale
        # below rather than baked into the image itself.
        depth_msg = self.bridge.cv2_to_imgmsg(frame.depth_image, encoding='16UC1')
        depth_msg.header = header

        info_msg = self.build_camera_info(frame.intrinsics, header)

        msg = RealsenseFrame()
        msg.header = header
        msg.rgb = rgb_msg
        msg.depth = depth_msg
        msg.camera_info = info_msg
        msg.depth_scale = float(frame.depth_scale)

        self.pub.publish(msg)

    def destroy_node(self):
        self.camera.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WristCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()