"""Save colour + depth + intrinsics from the running camera, on a keypress.

Every number the plug detector has been tuned against so far came from frames
ray-cast out of the CAD. That was enough to find real bugs -- a filled depth
halo, the side wall an oblique view adds to a silhouette, a rotated lattice
leaving a third of its cells empty -- but it cannot answer the question the
hardware keeps asking, which is how long the plug's outline actually comes out
when the moulding throws a highlight back at the lens. Measured off one frame it
was 15.8 mm against the CAD's 21.7. One frame is not enough to set a tolerance
from, and eyeballing a downscaled debug stream is how the plug got called white
when it is black.

So: capture a handful, commit them, and measure. That is what test_mono.py does
for the panel, on two frames rather than one, and the README says why -- every
regression that pipeline has had was invisible in the first.

    source install/setup.bash
    export ROS_LOCALHOST_ONLY=1
    ros2 launch py_gripper tmr_plug_launch.py     # or just the camera
    python3 tools/capture_frames.py --name plug   # in a second terminal

Enter saves a set, 'q' then Enter stops. Move the plug between shots: different
headings, different distances, and at least one with the cable curled back
alongside the head, that being the case the detector cannot currently solve and
the one worth having a real frame of.
"""
import argparse
import os
import sys
import threading

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'test_data')


class Capture(Node):
    def __init__(self, name, out_dir):
        super().__init__('capture_frames')
        self.name, self.out_dir = name, out_dir
        self.bgr = None
        self.depth = None
        self.K = None
        self.n = 0

        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', '/camera/camera/color/camera_info')

        self.create_subscription(Image, self.get_parameter('color_topic').value,
                                 self._color_cb, 1)
        self.create_subscription(Image, self.get_parameter('depth_topic').value,
                                 self._depth_cb, 1)
        self.create_subscription(CameraInfo, self.get_parameter('info_topic').value,
                                 self._info_cb, 10)

    def _color_cb(self, msg):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        self.bgr = np.ascontiguousarray(a[:, :, ::-1] if msg.encoding == 'rgb8' else a)

    def _depth_cb(self, msg):
        # Same decode the node uses, and saved in metres as float64 rather than
        # raw 16UC1, so a consumer cannot get the scale wrong later.
        if msg.encoding == '16UC1':
            raw = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
            d = raw.astype(np.float64) * 0.001
        elif msg.encoding == '32FC1':
            d = np.frombuffer(msg.data, np.float32).reshape(
                msg.height, msg.width).astype(np.float64)
        else:
            self.get_logger().error(f'unsupported depth encoding {msg.encoding}',
                                    throttle_duration_sec=10)
            return
        d[d <= 0] = np.nan
        self.depth = d

    def _info_cb(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def ready(self):
        missing = [n for n, v in (('colour', self.bgr), ('depth', self.depth),
                                  ('camera_info', self.K)) if v is None]
        return missing

    def save(self):
        missing = self.ready()
        if missing:
            print(f'  still waiting for: {", ".join(missing)}')
            return
        self.n += 1
        stem = os.path.join(self.out_dir, f'{self.name}{self.n:02d}')
        os.makedirs(self.out_dir, exist_ok=True)
        cv2.imwrite(f'{stem}_color.png', self.bgr)
        np.save(f'{stem}_depth.npy', self.depth.astype(np.float32))
        np.save(f'{stem}_K.npy', self.K)

        d = self.depth
        finite = np.isfinite(d)
        print(f'  saved {os.path.basename(stem)}_*  '
              f'{self.bgr.shape[1]}x{self.bgr.shape[0]}, '
              f'depth {finite.mean()*100:.0f}% valid, '
              f'median {np.nanmedian(d)*1000:.0f}mm, fx={self.K[0,0]:.0f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='plug', help='filename stem')
    ap.add_argument('--out', default=OUT_DIR, help='directory to write into')
    args = ap.parse_args()

    rclpy.init()
    node = Capture(args.name, args.out)
    # An executor rather than rclpy.spin in a daemon thread: spin has no way to
    # be told to stop, so the thread is still inside it when the main thread
    # tears the context down, and rclpy exits with `terminate called without an
    # active exception`. Nothing is lost either way, but a script whose normal
    # exit looks like a crash is a script people stop trusting.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    print(f'writing to {os.path.abspath(args.out)}/{args.name}NN_*')
    print('Enter to save a set, q + Enter to stop.\n')
    try:
        while True:
            if sys.stdin.readline().strip().lower() == 'q':
                break
            node.save()
    except KeyboardInterrupt:
        pass
    finally:
        print(f'\n{node.n} set(s) captured')
        executor.shutdown()
        spin.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
