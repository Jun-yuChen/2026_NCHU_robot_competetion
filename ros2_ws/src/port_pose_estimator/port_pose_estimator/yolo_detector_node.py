#!/usr/bin/env python3
"""
YoloDetectorNode
-------------------
Subscribes to RealsenseFrame (rgb + depth + camera_info + depth_scale),
runs YOLO detection on the color image, looks up a robust depth for
each bounding box, deprojects the bbox center to 3D camera coordinates
using the pinhole intrinsics from camera_info, and publishes the result
as a Detection3DArray.
"""

import torch
import rclpy
import cv2
from rclpy.node import Node
from cv_bridge import CvBridge

from geometry_msgs.msg import Pose

from custom_interface.msg import RealsenseFrame, Detection3D, Detection3DArray

from .utils.yolo_detector import YOLODetector
from .utils.camera import get_bbox_depth, Transform_pixel_to_camera


def camera_info_to_intrinsics(camera_info) -> dict:
    """
    Convert a sensor_msgs/CameraInfo.k (row-major 3x3, length-9) into the
    {"fx", "fy", "ppx", "ppy"} dict expected by Transform_pixel_to_camera.
    """
    K = camera_info.k
    return {
        "fx": K[0],
        "fy": K[4],
        "ppx": K[2],
        "ppy": K[5],
    }


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector_node")

        self.declare_parameter('yolo_model_path', '/path/to/0719_yolo.pt')
        yolo_model_path = self.get_parameter('yolo_model_path').value

        # Set to False (e.g. via a launch param later) to run headless.
        self.visualize = True
        self.window_name = "RGB-D + YOLO"

        self.bridge = CvBridge()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading YOLO detector {yolo_model_path} on device={device}")
        self.detector = YOLODetector(model_path=yolo_model_path, device=device)

        self.sub = self.create_subscription(
            RealsenseFrame,
            "realsense/frame",
            self.frame_callback,
            10,
        )

        self.pub = self.create_publisher(
            Detection3DArray,
            "yolo/detections_3d",
            10,
        )

        #self.get_logger().info("YoloDetectorNode ready.")

    def frame_callback(self, msg: RealsenseFrame):
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg.rgb, desired_encoding="bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(msg.depth, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        depth_scale = msg.depth_scale
        intrinsics = camera_info_to_intrinsics(msg.camera_info)

        detections = self.detector.detect(color_image, visualize=False)  # This node has its own visualization, disable visualization in detector. 

        out_msg = Detection3DArray()
        out_msg.header = msg.header

        vis = color_image.copy() if self.visualize else None

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class"]
            conf = det["conf"]

            #self.get_logger().info(f"object: {cls_name}")

            center_u = (x1 + x2) / 2.0
            center_v = (y1 + y2) / 2.0

            # get_bbox_depth expects bbox as (u1, v1, u2, v2), matching det["bbox"].
            depth_m = get_bbox_depth(det["bbox"], depth_image, depth_scale)

            if depth_m is not None:
                point_cam = Transform_pixel_to_camera(center_u, center_v, depth_m, intrinsics)
                x, y, z = float(point_cam[0]), float(point_cam[1]), float(point_cam[2])

                det3d = Detection3D()
                det3d.class_name = cls_name
                det3d.confidence = conf

                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                pose.orientation.w = 1.0  # identity; no orientation estimate available
                det3d.pose = pose

                out_msg.detections.append(det3d)

                coord_label = f"({x:.2f}, {y:.2f}, {z:.2f}) m"
            else:
                # No valid depth in this bbox -- skip publishing a pose for it,
                # but still draw it (marked N/A) so it's visible on screen.
                coord_label = "depth: N/A"

            if self.visualize:
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
                cv2.putText(
                    vis, f"{cls_name} {conf:.2f}", (p1[0], max(p1[1] - 25, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    vis, coord_label, (p1[0], max(p1[1] - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
                )

        self.pub.publish(out_msg)

        if self.visualize:
            cv2.imshow("RGB-D + YOLO", vis)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()