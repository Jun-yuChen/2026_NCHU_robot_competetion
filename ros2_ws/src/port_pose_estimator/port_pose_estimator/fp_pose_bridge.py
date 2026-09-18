"""
Bridges FoundationPose's live pose output into ROS2.

FoundationPose runs in its own micromamba/conda environment (GPU-heavy deps,
CUDA toolkit, pytorch3d, nvdiffrast) that is not ROS2-aware. This node stays
on the system ROS2 side: it opens a local TCP socket, accepts newline-delimited
JSON pose updates from a separate FoundationPose process (see
~/FoundationPose/track_and_publish.py), and republishes them as a ROS2
PoseStamped topic + TF transform so arm_cmd.py (or anything else) can consume
them normally.

Also converts the camera-frame pose into the robot base frame using the
measured hand-eye calibration (config/ICA_Lab_UMI_Config.yaml), following the
same world -> arm -> end_effector -> camera chain established in
arm_feedback_states.py: T_world_object = T_world_arm(live) @ T_G_C @ T_C_object.
"""
import json
import os
import socket
import threading
import time

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from tm_msgs.msg import FeedbackState

HOST = '127.0.0.1'
PORT = 9999

CALIB_PATH = os.path.join(get_package_share_directory('py_gripper'), 'calibration', 'ICA_Lab_UMI_Config.yaml')


def load_T_G_C():
    with open(CALIB_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return np.array(config['T_G_C'])


# TM5-900 reach is ~0.9m, so anything past this is not a real tool pose.
TOOL_POSE_MAX_ABS_XYZ = 2.0

# feedback_states normally arrives far faster than this. A *plausible but stale*
# tool_pose is more dangerous than an obviously garbage one: tool_pose_is_sane()
# happily passes it, and every world-frame pose then gets composed against where
# the arm used to be, so the computed object position is off by however far the
# arm has travelled since. Observed on hardware after a killed tm_driver left a
# zombie DDS publisher behind -- the subscription stayed matched to the dead
# endpoint, T_world_arm froze, and each move sent the arm further astray.
FEEDBACK_MAX_AGE_S = 0.5


def tool_pose_is_sane(tool_pose):
    # tm_driver caches the Ethernet-Slave field layout from the *first* packet
    # and afterwards copies by byte offset without re-checking field names, so a
    # layout mismatch silently shifts every field and yields garbage (observed:
    # [4.2e-44, 289590.272, 4.6e-44, 5054312.7, ...]). That garbage reached a
    # motion command once; only the TM controller's own range check stopped it.
    vals = list(tool_pose)
    if len(vals) != 6:
        return False
    if not all(np.isfinite(v) for v in vals):
        return False
    return all(abs(v) <= TOOL_POSE_MAX_ABS_XYZ for v in vals[:3])


def tool_pose_to_matrix(tool_pose):
    # TM feedback_states.tool_pose is [x, y, z, rx, ry, rz] (meters, radians, XYZ euler)
    T = np.eye(4)
    T[:3, 3] = tool_pose[:3]
    T[:3, :3] = Rotation.from_euler('xyz', tool_pose[3:]).as_matrix()
    return T


def rotmat_to_quat(R):
    # standard Shepperd's method, avoids adding a new dependency for this
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


class FpPoseBridge(Node):
    def __init__(self):
        super().__init__('fp_pose_bridge')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('object_frame', 'object')
        self.camera_frame = self.get_parameter('camera_frame').value
        self.object_frame = self.get_parameter('object_frame').value

        self.pose_pub = self.create_publisher(PoseStamped, 'camera_frame/object_pose', 10)
        self.world_pose_pub = self.create_publisher(PoseStamped, 'world_frame/object_pose', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.T_G_C = load_T_G_C()
        self.get_logger().info(f'loaded hand-eye calibration from {CALIB_PATH}')
        self.T_world_arm = None  # latest live tool_pose (feedback_states), set once known
        self.T_world_arm_stamp = None  # time.monotonic() when it was last refreshed
        self.feedback_sub = self.create_subscription(FeedbackState, 'feedback_states', self._feedback_cb, 10)

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        self.get_logger().info(f'Waiting for FoundationPose client on {HOST}:{PORT} ...')

        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.accept_thread.start()

    def _feedback_cb(self, msg):
        if not tool_pose_is_sane(msg.tool_pose):
            self.get_logger().error(
                f'tool_pose from tm_driver is garbage ({list(msg.tool_pose)}) -- refusing to use it. '
                'Restart tm_driver; if it persists, the TMflow Ethernet Slave data table changed.',
                throttle_duration_sec=5)
            self.T_world_arm = None
            self.T_world_arm_stamp = None
            return
        self.T_world_arm = tool_pose_to_matrix(msg.tool_pose)
        self.T_world_arm_stamp = time.monotonic()

    def _accept_loop(self):
        while rclpy.ok():
            conn, addr = self.server.accept()
            self.get_logger().info(f'FoundationPose client connected from {addr}')
            threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()

    def _client_loop(self, conn):
        f = conn.makefile('r', encoding='utf-8')
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._publish_pose(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    self.get_logger().warn(f'bad pose message: {e}')
        finally:
            conn.close()
            self.get_logger().info('FoundationPose client disconnected')

    def _publish_pose(self, msg):
        pose_4x4 = np.array(msg['pose'], dtype=np.float64).reshape(4, 4)
        t = pose_4x4[:3, 3]
        qx, qy, qz, qw = rotmat_to_quat(pose_4x4[:3, :3])

        stamp = self.get_clock().now().to_msg()

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.camera_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = t.tolist()
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pose_pub.publish(ps)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.camera_frame
        tf_msg.child_frame_id = self.object_frame
        tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z = t.tolist()
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf_msg)

        if self.T_world_arm is None:
            self.get_logger().warn('no feedback_states received yet, skipping world-frame pose', throttle_duration_sec=5)
            return

        age = time.monotonic() - self.T_world_arm_stamp
        if age > FEEDBACK_MAX_AGE_S:
            self.get_logger().error(
                f'feedback_states is {age:.1f}s stale (limit {FEEDBACK_MAX_AGE_S}s) -- refusing to '
                'publish world-frame pose, it would be composed against a stale arm pose. '
                'Check that tm_driver is alive and that no zombie tm_driver_node is still in the '
                'ROS graph (ros2 node list / ros2 topic info /feedback_states -v).',
                throttle_duration_sec=5)
            return

        # world -> arm(=G, TM's tool_pose) -> object, via T_G_C @ T_C_object
        T_world_object = self.T_world_arm @ self.T_G_C @ pose_4x4
        wt = T_world_object[:3, 3]
        wqx, wqy, wqz, wqw = rotmat_to_quat(T_world_object[:3, :3])

        wps = PoseStamped()
        wps.header.stamp = stamp
        wps.header.frame_id = 'world'
        wps.pose.position.x, wps.pose.position.y, wps.pose.position.z = wt.tolist()
        wps.pose.orientation.x = wqx
        wps.pose.orientation.y = wqy
        wps.pose.orientation.z = wqz
        wps.pose.orientation.w = wqw
        self.world_pose_pub.publish(wps)

        wtf = TransformStamped()
        wtf.header.stamp = stamp
        wtf.header.frame_id = 'world'
        wtf.child_frame_id = self.object_frame
        wtf.transform.translation.x, wtf.transform.translation.y, wtf.transform.translation.z = wt.tolist()
        wtf.transform.rotation.x = wqx
        wtf.transform.rotation.y = wqy
        wtf.transform.rotation.z = wqz
        wtf.transform.rotation.w = wqw
        self.tf_broadcaster.sendTransform(wtf)


def main(args=None):
    rclpy.init(args=args)
    node = FpPoseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
