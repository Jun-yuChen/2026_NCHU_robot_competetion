#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AprilTag 3-view teach/localization tool.

Stage 1 workflow:
  1. Human manually moves robot to a good overhead pose.
  2. `record` saves the current TM tool pose as the center/anchor pose.
  3. `localize` moves:
       center -> camera-left -> camera-right -> center
     while keeping the tool orientation unchanged.
  4. At center/left/right, detect the AprilTag with Eye-in-Hand camera.
  5. Compute:
       T_Base_Tag = T_Base_G @ T_G_C @ T_C_Tag
  6. Average the 3 estimates and save:
       T_Base_AprilTag
       T_AprilTag_Base

This script intentionally follows the transform convention used by the
provided eye_to_hand_calib_static_tag.py.

IMPORTANT:
- Tool pose rotation is interpreted exactly like the provided calibration
  script: scipy Rotation.from_euler('xyz', tool_pose[3:]).
- Left/right means camera-image left/right:
    optical +X = image right
    left  = -camera X
    right = +camera X
- Robot orientation is not changed during left/right teaching moves.
"""

import argparse
import math
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, JointState
from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SetPositions


PANEL_DIR = Path(__file__).resolve().parent
DATA_DIR = PANEL_DIR / "data"
DEFAULT_CONFIG = DATA_DIR / "apriltag_teach.yaml"
OBSERVE_POSE_FILE = DATA_DIR / "obeserve_pose.yaml"
FINAL_TRANSFORM_FILE = DATA_DIR / "observe_final_pose_transform.yaml"
THREE_VIEW_IMAGE_DIR = DATA_DIR / "apriltag_three_view_images"


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def mean_T(T_list):
    if not T_list:
        raise RuntimeError("mean_T 沒有樣本")
    mean_t = np.mean([T[:3, 3] for T in T_list], axis=0)
    mean_rotvec = np.mean(
        R.from_matrix([T[:3, :3] for T in T_list]).as_rotvec(),
        axis=0,
    )
    T_mean = np.eye(4, dtype=np.float64)
    T_mean[:3, :3] = R.from_rotvec(mean_rotvec).as_matrix()
    T_mean[:3, 3] = mean_t
    return T_mean


def pose_to_T(tool_pose):
    if len(tool_pose) < 6:
        raise RuntimeError("tool_pose 長度不足 6")
    values = np.asarray(tool_pose[:6], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("tool_pose 含 NaN/Inf")

    T = np.eye(4, dtype=np.float64)
    # Follow user's calibration program exactly.
    T[:3, :3] = R.from_euler(
        "xyz",
        values[3:6],
        degrees=False,
    ).as_matrix()
    T[:3, 3] = values[:3]
    return T


def T_to_pose(T):
    """4x4 Base->Tool matrix to TM [x,y,z,rx,ry,rz] using same XYZ Euler convention."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise RuntimeError(f"T_to_pose 需要 4x4，實際={T.shape}")
    euler = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)
    return [
        float(T[0, 3]),
        float(T[1, 3]),
        float(T[2, 3]),
        float(euler[0]),
        float(euler[1]),
        float(euler[2]),
    ]


def rotation_record(T):
    """Store camera orientation in both Euler XYZ and quaternion xyzw."""
    rot = R.from_matrix(np.asarray(T, dtype=np.float64)[:3, :3])
    return {
        "euler_xyz_rad": [float(v) for v in rot.as_euler("xyz", degrees=False)],
        "quaternion_xyzw": [float(v) for v in rot.as_quat()],
    }


def save_yaml_fixed(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_yaml_fixed(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"找不到紀錄檔：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"紀錄檔格式錯誤：{path}")
    return data


def save_three_view_image(name, frame):
    """Overwrite fixed CENTER/LEFT/RIGHT snapshots from the latest localization run."""
    if frame is None:
        return None
    import cv2
    THREE_VIEW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    filename = THREE_VIEW_IMAGE_DIR / f"{str(name).lower()}.jpg"
    ok = cv2.imwrite(str(filename), frame)
    if not ok:
        raise RuntimeError(f"無法寫入三視角照片：{filename}")
    return str(filename)


def clear_three_view_images():
    THREE_VIEW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("center.jpg", "left.jpg", "right.jpg"):
        p = THREE_VIEW_IMAGE_DIR / name
        if p.exists():
            p.unlink()


def load_matrix(yaml_path, key, invert=False):
    path = Path(os.path.expanduser(str(yaml_path))).resolve()
    if not path.is_file():
        raise RuntimeError(f"找不到 transform YAML: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or key not in data:
        raise RuntimeError(
            f"YAML 找不到 key '{key}'；"
            f"可用 keys={list(data.keys()) if isinstance(data, dict) else []}"
        )

    T = np.asarray(data[key], dtype=np.float64)
    if T.shape != (4, 4):
        raise RuntimeError(f"{key} 不是 4x4 matrix: {T.shape}")
    if not np.all(np.isfinite(T)):
        raise RuntimeError(f"{key} 含 NaN/Inf")

    if invert:
        T = invert_T(T)

    return T


class AprilTagTeachNode(Node):
    def __init__(
        self,
        tag_size,
        tag_id,
        image_topic,
        info_topic,
    ):
        super().__init__("apriltag_three_view_teach")

        self.current_tool_pose = None
        self.current_joint_names = None
        self.current_joint_positions = None
        self.info = None
        self.latest_image = None

        self.detector_backend = None
        self.detector = None
        self.cv2 = None
        self._init_apriltag_backend()

        self.tag_size = float(tag_size)
        self.tag_id = int(tag_id)

        self.pos_cli = self.create_client(
            SetPositions,
            "/set_positions",
        )

        self.create_subscription(
            FeedbackState,
            "/feedback_states",
            self.feedback_cb,
            10,
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_cb,
            10,
        )

        self.create_subscription(
            CameraInfo,
            info_topic,
            self.info_cb,
            10,
        )

        self.create_subscription(
            Image,
            image_topic,
            self.image_cb,
            10,
        )

    def _init_apriltag_backend(self):
        try:
            from dt_apriltags import Detector as DTDetector
            self.detector = DTDetector(
                searchpath=["apriltags"],
                families="tag36h11",
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            )
            self.detector_backend = "dt_apriltags"
            self.get_logger().info("AprilTag backend: dt_apriltags")
            return
        except Exception as error:
            self.get_logger().warn(
                f"dt_apriltags unavailable: {error}"
            )

        try:
            from pupil_apriltags import Detector as PupilDetector
            self.detector = PupilDetector(
                families="tag36h11",
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            )
            self.detector_backend = "pupil_apriltags"
            self.get_logger().info("AprilTag backend: pupil_apriltags")
            return
        except Exception as error:
            self.get_logger().warn(
                f"pupil_apriltags unavailable: {error}"
            )

        try:
            import cv2
            if not hasattr(cv2, "aruco"):
                raise RuntimeError("cv2 沒有 aruco module")
            if not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
                raise RuntimeError(
                    "cv2.aruco 沒有 DICT_APRILTAG_36h11"
                )

            self.cv2 = cv2
            dictionary = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_APRILTAG_36h11
            )
            if hasattr(cv2.aruco, "ArucoDetector"):
                params = cv2.aruco.DetectorParameters()
                self.detector = cv2.aruco.ArucoDetector(
                    dictionary,
                    params,
                )
            else:
                self.detector = dictionary

            self.detector_backend = "opencv_aruco"
            self.get_logger().info("AprilTag backend: OpenCV aruco")
            return
        except Exception as error:
            self.get_logger().warn(
                f"OpenCV AprilTag backend unavailable: {error}"
            )

        raise RuntimeError(
            "找不到可用的 AprilTag detector。"
            "已嘗試 dt_apriltags / pupil_apriltags / OpenCV aruco。"
            "Panel 已不再依賴 Python cv_bridge。"
        )

    @staticmethod
    def _image_msg_to_numpy(msg):
        """Decode sensor_msgs/Image exactly like original Panel viewer."""
        import cv2
        import numpy as _np

        h = int(msg.height)
        w = int(msg.width)
        enc = str(msg.encoding).lower()

        if h <= 0 or w <= 0:
            raise RuntimeError(
                f"invalid image size: {w}x{h}"
            )

        channels = {
            "bgr8": 3,
            "rgb8": 3,
            "bgra8": 4,
            "rgba8": 4,
            "mono8": 1,
        }.get(enc)

        if channels is None:
            raise RuntimeError(
                f"unsupported RGB encoding: {msg.encoding}"
            )

        row_bytes = w * channels
        step = (
            int(msg.step)
            if int(msg.step) > 0
            else row_bytes
        )

        raw = _np.frombuffer(
            msg.data,
            dtype=_np.uint8,
        )

        if raw.size < h * step:
            raise RuntimeError(
                f"RGB buffer too small: "
                f"got {raw.size}, need {h * step}"
            )

        rows = raw[:h * step].reshape(
            h,
            step,
        )[:, :row_bytes]

        if channels == 1:
            return rows.reshape(
                h,
                w,
            ).copy()

        image = rows.reshape(
            h,
            w,
            channels,
        )

        if enc == "bgr8":
            return image.copy()

        if enc == "rgb8":
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        if enc == "bgra8":
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2BGR,
            )

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2BGR,
        )

    def feedback_cb(self, msg):
        self.current_tool_pose = list(msg.tool_pose)

    def joint_state_cb(self, msg):
        names = [str(v) for v in msg.name]
        positions = [float(v) for v in msg.position]

        if len(positions) < 6:
            return

        # Prefer canonical TM joint order when names are available.
        canonical = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]

        if all(name in names for name in canonical):
            index = {name: i for i, name in enumerate(names)}
            self.current_joint_names = canonical
            self.current_joint_positions = [
                positions[index[name]]
                for name in canonical
            ]
        else:
            # Fallback: preserve message order for the first six joints.
            self.current_joint_names = (
                names[:6]
                if len(names) >= 6
                else [f"joint_{i+1}" for i in range(6)]
            )
            self.current_joint_positions = positions[:6]

    def info_cb(self, msg):
        self.info = {
            "fx": float(msg.k[0]),
            "fy": float(msg.k[4]),
            "cx": float(msg.k[2]),
            "cy": float(msg.k[5]),
        }

    def image_cb(self, msg):
        try:
            self.latest_image = self._image_msg_to_numpy(msg)
        except Exception as error:
            self.get_logger().error(
                f"Image decode failed: {error}"
            )

    def spin_until_ready(
        self,
        need_camera=False,
        need_joints=False,
        timeout=10.0,
    ):
        start = time.monotonic()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            feedback_ok = self.current_tool_pose is not None
            joints_ok = self.current_joint_positions is not None
            camera_ok = (
                self.info is not None
                and self.latest_image is not None
            )

            if (
                feedback_ok
                and (joints_ok or not need_joints)
                and (camera_ok or not need_camera)
            ):
                return

            if time.monotonic() - start > timeout:
                missing = []
                if not feedback_ok:
                    missing.append("/feedback_states")
                if need_joints and not joints_ok:
                    missing.append("/joint_states")
                if need_camera and self.info is None:
                    missing.append("camera_info")
                if need_camera and self.latest_image is None:
                    missing.append("camera image")

                if "/feedback_states" in missing:
                    raise RuntimeError(
                        "等不到 /feedback_states。"
                        "請先啟動 Panel 的「TM5-900 真機 / TM Driver」，"
                        "並確認 ros2 topic echo /feedback_states --once 有資料。"
                    )

                if "/joint_states" in missing:
                    raise RuntimeError(
                        "等不到 /joint_states。"
                        "PTP_J 示教需要六軸 joint positions；"
                        "請確認 TM Driver 正在發布 /joint_states。"
                    )

                raise RuntimeError(
                    "等待資料逾時: " + ", ".join(missing)
                )

    def wait_service(self, timeout=10.0):
        start = time.monotonic()
        while not self.pos_cli.wait_for_service(timeout_sec=0.5):
            if time.monotonic() - start > timeout:
                raise RuntimeError("找不到 /set_positions")

    def send_pose(
        self,
        pose,
        velocity,
        acc_time,
        motion="LINE_T",
    ):
        self.wait_service()

        req = SetPositions.Request()

        motion_name = str(motion).upper().strip()
        if motion_name == "PTP_T":
            req.motion_type = SetPositions.Request.PTP_T
        elif motion_name == "LINE_T":
            req.motion_type = SetPositions.Request.LINE_T
        else:
            raise RuntimeError(
                f"不支援 motion={motion}; 只支援 PTP_T / LINE_T"
            )

        req.positions = [float(v) for v in pose]
        req.velocity = float(velocity)
        req.acc_time = float(acc_time)
        req.blend_percentage = 0
        req.fine_goal = True

        self.get_logger().info(
            f"{motion_name} target = "
            + np.array2string(
                np.asarray(pose),
                precision=6,
                separator=", ",
            )
        )

        future = self.pos_cli.call_async(req)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=10.0,
        )

        if not future.done():
            raise RuntimeError("/set_positions call timeout")

        result = future.result()
        if result is None:
            raise RuntimeError("/set_positions 沒有回應")

        self.get_logger().info(
            f"/set_positions response = {result}"
        )

        if hasattr(result, "ok") and not bool(result.ok):
            raise RuntimeError(
                f"/set_positions rejected: {result}"
            )

        return result

    def wait_arrived(
        self,
        target,
        pos_error=0.004,
        rot_error_deg=1.0,
        timeout=30.0,
    ):
        target = np.asarray(target, dtype=np.float64)
        start = time.monotonic()
        last_log = 0.0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.current_tool_pose is None:
                continue

            current = np.asarray(
                self.current_tool_pose[:6],
                dtype=np.float64,
            )

            pos_diff = float(
                np.linalg.norm(target[:3] - current[:3])
            )

            R_target = R.from_euler(
                "xyz",
                target[3:],
                degrees=False,
            )
            R_current = R.from_euler(
                "xyz",
                current[3:],
                degrees=False,
            )

            rot_diff_deg = (
                (R_target.inv() * R_current).magnitude()
                * 180.0
                / math.pi
            )

            now = time.monotonic()
            if now - last_log >= 1.0:
                self.get_logger().info(
                    "waiting arrival: "
                    f"pos_error={pos_diff:.4f} m, "
                    f"rot_error={rot_diff_deg:.2f} deg\n"
                    f"  current={np.array2string(current, precision=6, separator=', ')}\n"
                    f"  target ={np.array2string(target, precision=6, separator=', ')}"
                )
                last_log = now

            if (
                pos_diff <= pos_error
                and rot_diff_deg <= rot_error_deg
            ):
                self.get_logger().info(
                    "arrived: "
                    f"pos_error={pos_diff:.4f} m, "
                    f"rot_error={rot_diff_deg:.2f} deg"
                )
                return

            if now - start > timeout:
                raise RuntimeError(
                    "等待機器人到位逾時；"
                    f"pos_error={pos_diff:.4f}m, "
                    f"rot_error={rot_diff_deg:.2f}deg；"
                    f"current={current.tolist()}；"
                    f"target={target.tolist()}"
                )

    def send_joint_pose(
        self,
        joint_positions,
        velocity=0.20,
        acc_time=0.5,
    ):
        """Send a PTP_J command. Joint positions are radians."""
        self.wait_service()

        joints = np.asarray(
            joint_positions,
            dtype=np.float64,
        ).reshape(-1)

        if joints.size != 6:
            raise RuntimeError(
                f"PTP_J 需要 6 個 joint angles，實際={joints.size}"
            )
        if not np.all(np.isfinite(joints)):
            raise RuntimeError("PTP_J joint angles 含 NaN/Inf")

        req = SetPositions.Request()
        req.motion_type = SetPositions.Request.PTP_J
        req.positions = [float(v) for v in joints]
        req.velocity = float(velocity)
        req.acc_time = float(acc_time)
        req.blend_percentage = 0
        req.fine_goal = True

        self.get_logger().info(
            "PTP_J target [rad] = "
            + np.array2string(
                joints,
                precision=7,
                separator=", ",
            )
        )

        future = self.pos_cli.call_async(req)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=10.0,
        )

        if not future.done():
            raise RuntimeError("/set_positions PTP_J call timeout")

        result = future.result()
        if result is None:
            raise RuntimeError("/set_positions PTP_J 沒有回應")

        self.get_logger().info(
            f"/set_positions PTP_J response = {result}"
        )

        if hasattr(result, "ok") and not bool(result.ok):
            raise RuntimeError(
                f"/set_positions PTP_J rejected: {result}"
            )

        return result

    def wait_joint_arrived(
        self,
        target_joints,
        joint_error_rad=0.02,
        timeout=30.0,
    ):
        target = np.asarray(
            target_joints,
            dtype=np.float64,
        ).reshape(6)

        start = time.monotonic()
        last_log = 0.0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.current_joint_positions is None:
                continue

            current = np.asarray(
                self.current_joint_positions,
                dtype=np.float64,
            ).reshape(6)

            errors = np.abs(target - current)
            max_error = float(np.max(errors))

            now = time.monotonic()
            if now - last_log >= 1.0:
                self.get_logger().info(
                    "waiting PTP_J arrival:\n"
                    f"  current={np.array2string(current, precision=6, separator=', ')}\n"
                    f"  target ={np.array2string(target, precision=6, separator=', ')}\n"
                    f"  abs_err={np.array2string(errors, precision=5, separator=', ')} rad"
                )
                last_log = now

            if max_error <= float(joint_error_rad):
                self.get_logger().info(
                    f"PTP_J arrived: max_joint_error={max_error:.5f} rad"
                )
                return

            if now - start > timeout:
                raise RuntimeError(
                    "等待 PTP_J 到位逾時；"
                    f"max_joint_error={max_error:.5f} rad；"
                    f"current={current.tolist()}；"
                    f"target={target.tolist()}"
                )

    def settle(self, seconds=1.0):
        end = time.monotonic() + float(seconds)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.03)

    def detect_tag_once(self):
        if self.latest_image is None or self.info is None:
            return None

        frame = self.latest_image.copy()
        if frame.ndim == 2:
            gray = frame
        else:
            b = frame[:, :, 0].astype(np.uint16)
            g = frame[:, :, 1].astype(np.uint16)
            r = frame[:, :, 2].astype(np.uint16)
            gray = ((29 * b + 150 * g + 77 * r) >> 8).astype(
                np.uint8
            )

        cam_params = [
            self.info["fx"],
            self.info["fy"],
            self.info["cx"],
            self.info["cy"],
        ]

        if self.detector_backend in (
            "dt_apriltags",
            "pupil_apriltags",
        ):
            results = self.detector.detect(
                gray,
                True,
                cam_params,
                self.tag_size,
            )
            if not results:
                return None

            selected = None
            for result in results:
                if int(result.tag_id) == self.tag_id:
                    selected = result
                    break

            if selected is None:
                ids = [int(r.tag_id) for r in results]
                self.get_logger().warn(
                    f"看到 AprilTag ids={ids}，"
                    f"但找不到指定 tag_id={self.tag_id}"
                )
                return None

            T_C_Tag = np.eye(4, dtype=np.float64)
            T_C_Tag[:3, :3] = selected.pose_R
            T_C_Tag[:3, 3] = selected.pose_t.reshape(3)
            return T_C_Tag

        if self.detector_backend == "opencv_aruco":
            cv2 = self.cv2

            if hasattr(self.detector, "detectMarkers"):
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray,
                    self.detector,
                )

            if ids is None or len(ids) == 0:
                return None

            ids_flat = ids.reshape(-1).astype(int)
            match = np.where(ids_flat == self.tag_id)[0]
            if len(match) == 0:
                self.get_logger().warn(
                    f"看到 AprilTag ids={ids_flat.tolist()}，"
                    f"但找不到指定 tag_id={self.tag_id}"
                )
                return None

            idx = int(match[0])
            image_points = np.asarray(
                corners[idx],
                dtype=np.float64,
            ).reshape(4, 2)

            s = float(self.tag_size) / 2.0
            object_points = np.array(
                [
                    [-s,  s, 0.0],
                    [ s,  s, 0.0],
                    [ s, -s, 0.0],
                    [-s, -s, 0.0],
                ],
                dtype=np.float64,
            )

            camera_matrix = np.array(
                [
                    [self.info["fx"], 0.0, self.info["cx"]],
                    [0.0, self.info["fy"], self.info["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            dist_coeffs = np.zeros((5, 1), dtype=np.float64)

            flags = (
                cv2.SOLVEPNP_IPPE_SQUARE
                if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
                else cv2.SOLVEPNP_ITERATIVE
            )

            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=flags,
            )
            if not ok:
                return None

            rot, _ = cv2.Rodrigues(rvec)
            T_C_Tag = np.eye(4, dtype=np.float64)
            T_C_Tag[:3, :3] = rot
            T_C_Tag[:3, 3] = tvec.reshape(3)
            return T_C_Tag

        return None

    def detect_tag_stable(
        self,
        count=5,
        timeout=5.0,
    ):
        samples = []
        start = time.monotonic()
        last_capture = 0.0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.03)

            now = time.monotonic()
            if now - last_capture >= 0.12:
                T = self.detect_tag_once()
                last_capture = now
                if T is not None:
                    samples.append(T)
                    self.get_logger().info(
                        f"Tag sample {len(samples)}/{count}: "
                        f"camera distance="
                        f"{np.linalg.norm(T[:3,3]):.4f} m"
                    )

            if len(samples) >= count:
                return mean_T(samples)

            if now - start > timeout:
                raise RuntimeError(
                    f"AprilTag 穩定取樣不足："
                    f"{len(samples)}/{count}"
                )


def load_teach_config(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_teach_config(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def record_anchor(node, config_path, T_G_C):
    """Record the manually taught overhead observation pose.

    Stores:
    - TM Cartesian tool pose
    - the actual six joint angles from /joint_states
    - Eye-in-Hand camera pose/orientation
    """
    node.spin_until_ready(
        need_camera=False,
        need_joints=True,
    )

    anchor = [float(v) for v in node.current_tool_pose[:6]]
    anchor_joints = [
        float(v)
        for v in node.current_joint_positions[:6]
    ]
    T_Base_G = pose_to_T(anchor)
    T_Base_Camera = T_Base_G @ T_G_C

    observe_data = {
        "name": "obeserve_pose",
        "saved_at_unix": float(time.time()),
        "tool_pose_xyz_rpy_rad": anchor,
        "joint_names": list(node.current_joint_names or []),
        "joint_positions_rad": anchor_joints,
        "T_Base_Tool": T_Base_G.tolist(),
        "T_Base_Camera": T_Base_Camera.tolist(),
        "camera_position_base_m": [float(v) for v in T_Base_Camera[:3, 3]],
        "camera_orientation": rotation_record(T_Base_Camera),
    }
    save_yaml_fixed(OBSERVE_POSE_FILE, observe_data)

    # Keep apriltag_teach.yaml synchronized because the 3-view locator uses this anchor.
    data = load_teach_config(config_path)
    data["apriltag_search_anchor_tool_pose"] = anchor
    data["apriltag_search_anchor_saved_at_unix"] = float(time.time())
    data["observe_pose_file"] = str(OBSERVE_POSE_FILE)
    save_teach_config(config_path, data)

    print("\n========================================")
    print("OVERHEAD OBSERVE POSE SAVED")
    print("========================================")
    print("tool_pose [x,y,z,rx,ry,rz]:")
    print(np.array2string(np.asarray(anchor), precision=8, separator=", "))
    print("")
    print("joint_positions [rad]:")
    print(np.array2string(
        np.asarray(anchor_joints),
        precision=8,
        separator=", ",
    ))
    print("")
    print("camera orientation quaternion [x,y,z,w]:")
    print(np.array2string(
        np.asarray(observe_data["camera_orientation"]["quaternion_xyzw"]),
        precision=8,
        separator=", ",
    ))
    print("")
    print("fixed file:")
    print(OBSERVE_POSE_FILE)
    print("========================================")


def localize_three_views(
    node,
    config_path,
    T_G_C,
    offset,
    velocity,
    acc_time,
    samples_per_view,
    return_to_center=True,
):
    if offset <= 0.0 or offset > 0.15:
        raise RuntimeError(
            "左右平移量必須 > 0 且 <= 0.15 m"
        )

    data = load_teach_config(config_path)
    anchor = data.get(
        "apriltag_search_anchor_tool_pose"
    )

    if (
        not isinstance(anchor, list)
        or len(anchor) != 6
    ):
        raise RuntimeError(
            "尚未記錄中央姿態。"
            "請先手動調好俯視姿態，再按「記錄目前中央姿態」。"
        )

    anchor = np.asarray(anchor, dtype=np.float64)

    node.spin_until_ready(
        need_camera=True,
        timeout=12.0,
    )

    # Center orientation, exactly following the calibration chain.
    T_B_G_anchor = pose_to_T(anchor)
    T_B_C_anchor = T_B_G_anchor @ T_G_C

    # Optical +X is image right.
    camera_right_base = T_B_C_anchor[:3, 0].copy()
    camera_right_base /= np.linalg.norm(camera_right_base)

    center = anchor.copy()

    left = anchor.copy()
    left[:3] = (
        anchor[:3]
        - camera_right_base * offset
    )

    right = anchor.copy()
    right[:3] = (
        anchor[:3]
        + camera_right_base * offset
    )

    poses = [
        ("CENTER", center),
        ("LEFT", left),
        ("RIGHT", right),
    ]

    T_B_Tag_list = []
    view_results = []
    clear_three_view_images()

    try:
        for name, target_pose in poses:
            print("\n----------------------------------------")
            print(f"VIEW = {name}")
            print("----------------------------------------")

            node.send_pose(
                target_pose.tolist(),
                velocity=velocity,
                acc_time=acc_time,
                motion="LINE_T",
            )
            node.wait_arrived(target_pose)
            node.settle(1.0)

            # Use actual feedback pose, not commanded pose.
            actual_pose = [
                float(v)
                for v in node.current_tool_pose[:6]
            ]
            actual_joints = (
                [float(v) for v in node.current_joint_positions[:6]]
                if node.current_joint_positions is not None
                else None
            )
            actual_joint_names = (
                list(node.current_joint_names)
                if node.current_joint_names is not None
                else None
            )
            T_B_G = pose_to_T(actual_pose)

            T_C_Tag = node.detect_tag_stable(
                count=samples_per_view,
                timeout=6.0,
            )

            snapshot_path = save_three_view_image(
                name,
                None if node.latest_image is None else node.latest_image.copy(),
            )

            T_B_Tag = (
                T_B_G
                @ T_G_C
                @ T_C_Tag
            )

            T_B_Tag_list.append(T_B_Tag)
            view_results.append({
                "name": name,
                "tool_pose": actual_pose,
                "joint_names": actual_joint_names,
                "joint_positions_rad": actual_joints,
                "T_Camera_AprilTag": T_C_Tag.tolist(),
                "T_Base_AprilTag": T_B_Tag.tolist(),
                "snapshot_path": snapshot_path,
            })

            print(
                f"{name} T_Base_AprilTag translation = "
                f"{np.array2string(T_B_Tag[:3,3], precision=6)}"
            )

        T_B_Tag_mean = mean_T(T_B_Tag_list)
        T_Tag_B = invert_T(T_B_Tag_mean)

        pos_errors = np.asarray([
            T[:3, 3] - T_B_Tag_mean[:3, 3]
            for T in T_B_Tag_list
        ])

        rot_errors_deg = []
        R_mean = R.from_matrix(
            T_B_Tag_mean[:3, :3]
        )
        for T in T_B_Tag_list:
            Ri = R.from_matrix(T[:3, :3])
            rot_errors_deg.append(
                (R_mean.inv() * Ri).magnitude()
                * 180.0
                / math.pi
            )

        data["apriltag_tag_id"] = int(node.tag_id)
        data["apriltag_tag_size_m"] = float(node.tag_size)
        data["apriltag_lateral_offset_m"] = float(offset)
        data["apriltag_camera_right_axis_base"] = (
            camera_right_base.tolist()
        )
        data["apriltag_three_view_samples"] = view_results
        data["apriltag_three_view_image_dir"] = str(THREE_VIEW_IMAGE_DIR)
        data["T_Base_AprilTag"] = T_B_Tag_mean.tolist()
        data["T_AprilTag_Base"] = T_Tag_B.tolist()
        data["apriltag_position_error_each_m"] = (
            pos_errors.tolist()
        )
        data["apriltag_position_std_m"] = (
            np.std(pos_errors, axis=0).tolist()
        )
        data["apriltag_rotation_error_each_deg"] = [
            float(v) for v in rot_errors_deg
        ]
        data["apriltag_localized_at_unix"] = (
            float(time.time())
        )

        save_teach_config(config_path, data)

        print("\n========================================")
        print("APRILTAG 3-VIEW LOCALIZATION COMPLETE")
        print("========================================")
        print("T_Base_AprilTag:")
        print(np.array2string(
            T_B_Tag_mean,
            precision=8,
            separator=", ",
        ))
        print("")
        print(
            "position std [m] =",
            np.array2string(
                np.std(pos_errors, axis=0),
                precision=6,
            ),
        )
        print(
            "rotation error [deg] =",
            np.array2string(
                np.asarray(rot_errors_deg),
                precision=4,
            ),
        )
        print("")
        print("saved to:")
        print(config_path)
        print("========================================")

    finally:
        if return_to_center:
            try:
                print("\nReturning to CENTER anchor...")
                node.send_pose(
                    center.tolist(),
                    velocity=velocity,
                    acc_time=acc_time,
                    motion="LINE_T",
                )
                node.wait_arrived(center)
                print("Returned to CENTER.")
            except Exception as error:
                print(
                    "[WARN] 回中央姿態失敗：",
                    error,
                )
        else:
            print("\n[FLOW] return_to_center=False：保留在最後 RIGHT 視角，直接銜接下一段流程。")




def record_final_pose(node, config_path, T_G_C):
    """Record the manually taught final observation relation to the latest AprilTag pose.

    Fixed output:
      ~/.tm_ros2_control_panel/observe_final_pose_transform.yaml

    Primary reusable transform:
      T_AprilTag_CameraFinal

    Later:
      T_Base_CameraGoal = T_Base_AprilTag(new) @ T_AprilTag_CameraFinal
      T_Base_ToolGoal   = T_Base_CameraGoal @ inv(T_G_C)
    """
    node.spin_until_ready(
        need_camera=False,
        need_joints=True,
    )

    data = load_teach_config(config_path)
    T_Base_Tag_raw = data.get("T_Base_AprilTag")
    if T_Base_Tag_raw is None:
        raise RuntimeError(
            "尚未有 T_Base_AprilTag。"
            "請先執行中央/左/右三視角定位，再手動移到最終觀察姿態後記錄。"
        )

    T_Base_Tag = np.asarray(T_Base_Tag_raw, dtype=np.float64)
    if T_Base_Tag.shape != (4, 4):
        raise RuntimeError("T_Base_AprilTag 格式不是 4x4")

    current_tool_pose = [float(v) for v in node.current_tool_pose[:6]]
    current_joint_positions = [
        float(v)
        for v in node.current_joint_positions[:6]
    ]
    current_joint_names = list(node.current_joint_names or [])
    T_Base_G = pose_to_T(current_tool_pose)
    T_Base_Camera = T_Base_G @ T_G_C

    T_Tag_CameraFinal = invert_T(T_Base_Tag) @ T_Base_Camera
    T_Tag_ToolFinal = invert_T(T_Base_Tag) @ T_Base_G

    final_data = {
        "name": "observe_final_pose_transform",
        "saved_at_unix": float(time.time()),
        "source_T_Base_AprilTag": T_Base_Tag.tolist(),
        "recorded_tool_pose_xyz_rpy_rad": current_tool_pose,
        "recorded_joint_names": current_joint_names,
        "recorded_joint_positions_rad": current_joint_positions,
        "recorded_T_Base_Tool": T_Base_G.tolist(),
        "recorded_T_Base_Camera": T_Base_Camera.tolist(),
        "recorded_camera_orientation_base": rotation_record(T_Base_Camera),
        "T_AprilTag_CameraFinal": T_Tag_CameraFinal.tolist(),
        "T_AprilTag_ToolFinal": T_Tag_ToolFinal.tolist(),
        "camera_position_relative_to_apriltag_m": [
            float(v) for v in T_Tag_CameraFinal[:3, 3]
        ],
        "camera_orientation_relative_to_apriltag": rotation_record(
            T_Tag_CameraFinal
        ),
    }
    save_yaml_fixed(FINAL_TRANSFORM_FILE, final_data)

    data["observe_final_pose_transform_file"] = str(FINAL_TRANSFORM_FILE)
    data["T_AprilTag_CameraFinal"] = T_Tag_CameraFinal.tolist()
    save_teach_config(config_path, data)

    print("\n========================================")
    print("FINAL OBSERVATION TRANSFORM SAVED")
    print("========================================")
    print("T_AprilTag_CameraFinal:")
    print(np.array2string(
        T_Tag_CameraFinal,
        precision=8,
        separator=", ",
    ))
    print("")
    print("camera XYZ relative to AprilTag [m]:")
    print(np.array2string(
        T_Tag_CameraFinal[:3, 3],
        precision=8,
        separator=", ",
    ))
    print("")
    print("recorded final joint_positions [rad]:")
    print(np.array2string(
        np.asarray(current_joint_positions),
        precision=8,
        separator=", ",
    ))
    print("")
    print("camera quaternion relative to AprilTag [x,y,z,w]:")
    print(np.array2string(
        np.asarray(
            final_data["camera_orientation_relative_to_apriltag"][
                "quaternion_xyzw"
            ]
        ),
        precision=8,
        separator=", ",
    ))
    print("")
    print("fixed file:")
    print(FINAL_TRANSFORM_FILE)
    print("========================================")



def load_observe_joint_target():
    observe_data = load_yaml_fixed(OBSERVE_POSE_FILE)
    joints = observe_data.get("joint_positions_rad")
    if not isinstance(joints, list) or len(joints) != 6:
        raise RuntimeError(
            f"{OBSERVE_POSE_FILE} 沒有 joint_positions_rad。"
            "這是舊版紀錄檔，請重新執行「記錄俯視姿態」後再測 PTP_J。"
        )
    return observe_data, [float(v) for v in joints]


def sync_anchor_from_observe(config_path):
    observe_data = load_yaml_fixed(OBSERVE_POSE_FILE)
    anchor = observe_data.get("tool_pose_xyz_rpy_rad")
    if not isinstance(anchor, list) or len(anchor) != 6:
        raise RuntimeError(
            f"{OBSERVE_POSE_FILE} 缺少 tool_pose_xyz_rpy_rad"
        )

    data = load_teach_config(config_path)
    data["apriltag_search_anchor_tool_pose"] = [
        float(v) for v in anchor
    ]
    data["apriltag_search_anchor_saved_at_unix"] = float(time.time())
    save_teach_config(config_path, data)
    return [float(v) for v in anchor]


def compute_latest_final_goal(config_path, T_G_C):
    final_data = load_yaml_fixed(FINAL_TRANSFORM_FILE)
    localized = load_teach_config(config_path)

    T_Base_Tag_raw = localized.get("T_Base_AprilTag")
    if T_Base_Tag_raw is None:
        raise RuntimeError(
            "目前 apriltag_teach.yaml 沒有 T_Base_AprilTag；"
            "請先做三視角定位。"
        )

    T_Tag_CameraFinal_raw = final_data.get(
        "T_AprilTag_CameraFinal"
    )
    if T_Tag_CameraFinal_raw is None:
        raise RuntimeError(
            f"{FINAL_TRANSFORM_FILE} 缺少 T_AprilTag_CameraFinal"
        )

    T_Base_Tag = np.asarray(
        T_Base_Tag_raw,
        dtype=np.float64,
    )
    T_Tag_CameraFinal = np.asarray(
        T_Tag_CameraFinal_raw,
        dtype=np.float64,
    )

    if T_Base_Tag.shape != (4, 4):
        raise RuntimeError("T_Base_AprilTag 不是 4x4")
    if T_Tag_CameraFinal.shape != (4, 4):
        raise RuntimeError("T_AprilTag_CameraFinal 不是 4x4")

    T_Base_CameraGoal = (
        T_Base_Tag @ T_Tag_CameraFinal
    )
    T_Base_ToolGoal = (
        T_Base_CameraGoal @ invert_T(T_G_C)
    )
    final_tool_pose = T_to_pose(T_Base_ToolGoal)

    return (
        T_Base_Tag,
        T_Base_CameraGoal,
        T_Base_ToolGoal,
        final_tool_pose,
    )


def step_check_data(node, config_path, T_G_C):
    node.spin_until_ready(
        need_camera=False,
        need_joints=True,
    )

    print("\n========================================")
    print("STEP TEST 0: DATA / CURRENT ROBOT STATE")
    print("========================================")
    print("observe pose file:", OBSERVE_POSE_FILE)
    print("exists:", OBSERVE_POSE_FILE.is_file())
    print("final transform file:", FINAL_TRANSFORM_FILE)
    print("exists:", FINAL_TRANSFORM_FILE.is_file())
    print("apriltag config:", config_path)
    print("exists:", config_path.is_file())
    print("")
    print("current tool pose:")
    print(np.array2string(
        np.asarray(node.current_tool_pose[:6]),
        precision=7,
        separator=", ",
    ))
    print("current joint names:")
    print(node.current_joint_names)
    print("current joint positions [rad]:")
    print(np.array2string(
        np.asarray(node.current_joint_positions[:6]),
        precision=7,
        separator=", ",
    ))

    if OBSERVE_POSE_FILE.is_file():
        observe_data = load_yaml_fixed(OBSERVE_POSE_FILE)
        print("")
        print("saved observe joints [rad]:")
        print(observe_data.get("joint_positions_rad"))
        print("saved observe tool pose:")
        print(observe_data.get("tool_pose_xyz_rpy_rad"))

    print("========================================")


def step_move_observe_joint(
    node,
    joint_velocity,
    acc_time,
):
    _, joints = load_observe_joint_target()

    node.spin_until_ready(
        need_camera=False,
        need_joints=True,
    )

    print("\n========================================")
    print("STEP TEST 1: PTP_J -> OVERHEAD OBSERVE POSE")
    print("========================================")
    node.send_joint_pose(
        joints,
        velocity=joint_velocity,
        acc_time=acc_time,
    )
    node.wait_joint_arrived(joints)
    node.settle(0.5)
    print("STEP TEST 1 COMPLETE")
    print("========================================")



def step_move_recorded_final_joint(
    node,
    joint_velocity,
    acc_time,
):
    """Same-scene diagnostic: return exactly to the manually taught final joint configuration."""
    final_data = load_yaml_fixed(FINAL_TRANSFORM_FILE)

    joints = final_data.get("recorded_joint_positions_rad")
    if not isinstance(joints, list) or len(joints) != 6:
        raise RuntimeError(
            f"{FINAL_TRANSFORM_FILE} 沒有 recorded_joint_positions_rad。"
            "請用新版程式重新執行「記錄人工調整後的最終觀察姿態」。"
        )

    joints = [float(v) for v in joints]

    node.spin_until_ready(
        need_camera=False,
        need_joints=True,
    )

    print("\n========================================")
    print("STEP TEST: PTP_J -> MANUALLY TAUGHT FINAL POSE")
    print("========================================")
    print("recorded final joints [rad]:")
    print(np.array2string(
        np.asarray(joints),
        precision=8,
        separator=", ",
    ))
    print("")
    print("This test does NOT use AprilTag transform or Cartesian IK.")
    print("It only replays the exact joint configuration recorded during manual teaching.")
    print("")

    node.send_joint_pose(
        joints,
        velocity=joint_velocity,
        acc_time=acc_time,
    )
    node.wait_joint_arrived(joints)
    node.settle(0.5)

    print("PTP_J FINAL-POSE REPLAY COMPLETE")
    print("========================================")



def step_preview_final(node, config_path, T_G_C):
    (
        T_Base_Tag,
        T_Base_CameraGoal,
        T_Base_ToolGoal,
        final_tool_pose,
    ) = compute_latest_final_goal(
        config_path,
        T_G_C,
    )

    print("\n========================================")
    print("STEP TEST 3: PREVIEW FINAL TARGET (NO MOVE)")
    print("========================================")
    print("T_Base_AprilTag:")
    print(np.array2string(
        T_Base_Tag,
        precision=8,
        separator=", ",
    ))
    print("")
    print("T_Base_CameraGoal:")
    print(np.array2string(
        T_Base_CameraGoal,
        precision=8,
        separator=", ",
    ))
    print("")
    print("T_Base_ToolGoal:")
    print(np.array2string(
        T_Base_ToolGoal,
        precision=8,
        separator=", ",
    ))
    print("")
    print("final TM tool target [x,y,z,rx,ry,rz]:")
    print(np.array2string(
        np.asarray(final_tool_pose),
        precision=8,
        separator=", ",
    ))
    print("")
    print("NO ROBOT MOTION WAS SENT.")
    print("========================================")



def pre_adjust_joint2(
    node,
    offset_deg,
    joint_velocity,
    acc_time,
):
    """Move only Joint2 by a signed offset before the final Cartesian PTP_T.

    Positive offset means: target Joint2 = current Joint2 + offset_deg.
    Use a negative value in the Panel if the physical direction is opposite.
    This intentionally changes the IK/planning start configuration before
    the final Cartesian target is submitted to the TM controller.
    """
    node.spin_until_ready(
        need_camera=False,
        need_joints=True,
    )

    current = np.asarray(
        node.current_joint_positions[:6],
        dtype=np.float64,
    )
    target = current.copy()
    offset_rad = math.radians(float(offset_deg))
    target[1] += offset_rad

    print("\n========================================")
    print("PRE-FINAL JOINT2 ADJUSTMENT")
    print("========================================")
    print(f"Joint2 signed offset = {float(offset_deg):.4f} deg ({offset_rad:.8f} rad)")
    print("positive value means current J2 + offset")
    print("current joints [rad]:")
    print(np.array2string(current, precision=8, separator=", "))
    print("pre-adjust target joints [rad]:")
    print(np.array2string(target, precision=8, separator=", "))

    node.send_joint_pose(
        target.tolist(),
        velocity=joint_velocity,
        acc_time=acc_time,
    )
    node.wait_joint_arrived(target)
    node.settle(0.4)

    print("Joint2 pre-adjustment complete.")
    print("The final PTP_T will now be submitted from this new joint configuration.")
    print("========================================")


def step_move_final(
    node,
    config_path,
    T_G_C,
    velocity,
    joint_velocity,
    joint2_pre_offset_deg,
    acc_time,
):
    (
        _,
        _,
        _,
        final_tool_pose,
    ) = compute_latest_final_goal(
        config_path,
        T_G_C,
    )

    node.spin_until_ready(
        need_camera=False,
        need_joints=True,
    )

    print("\n========================================")
    print("STEP TEST 4B: JOINT2 PRE-ADJUST -> FINAL PTP_T")
    print("========================================")
    print("PTP_T target:")
    print(np.array2string(
        np.asarray(final_tool_pose),
        precision=8,
        separator=", ",
    ))

    pre_adjust_joint2(
        node=node,
        offset_deg=joint2_pre_offset_deg,
        joint_velocity=joint_velocity,
        acc_time=acc_time,
    )

    print("Planning/executing final PTP_T from the adjusted Joint2 start state...")
    node.send_pose(
        final_tool_pose,
        velocity=velocity,
        acc_time=acc_time,
        motion="PTP_T",
    )
    node.wait_arrived(
        np.asarray(final_tool_pose, dtype=np.float64)
    )
    node.settle(0.5)
    print("STEP TEST 4 COMPLETE")
    print("========================================")



def run_full_workflow(
    node,
    config_path,
    T_G_C,
    offset,
    velocity,
    joint_velocity,
    joint2_pre_offset_deg,
    acc_time,
    samples_per_view,
):
    """One-click actual workflow.

    1) Load obeserve_pose.yaml and move to taught overhead pose.
    2) Run center/left/right AprilTag localization.
    3) Load observe_final_pose_transform.yaml.
    4) Reconstruct the final camera pose relative to the newly localized AprilTag.
    5) Convert camera goal back to TM tool goal and move there.
    """
    observe_data, observe_joints = load_observe_joint_target()
    final_data = load_yaml_fixed(FINAL_TRANSFORM_FILE)

    anchor = observe_data.get("tool_pose_xyz_rpy_rad")
    if not isinstance(anchor, list) or len(anchor) != 6:
        raise RuntimeError(
            f"{OBSERVE_POSE_FILE} 缺少 tool_pose_xyz_rpy_rad"
        )

    T_Tag_CameraFinal_raw = final_data.get("T_AprilTag_CameraFinal")
    if T_Tag_CameraFinal_raw is None:
        raise RuntimeError(
            f"{FINAL_TRANSFORM_FILE} 缺少 T_AprilTag_CameraFinal"
        )
    T_Tag_CameraFinal = np.asarray(
        T_Tag_CameraFinal_raw,
        dtype=np.float64,
    )
    if T_Tag_CameraFinal.shape != (4, 4):
        raise RuntimeError("T_AprilTag_CameraFinal 不是 4x4")

    node.spin_until_ready(
        need_camera=True,
        need_joints=True,
        timeout=12.0,
    )

    print("\n========================================")
    print("AUTO WORKFLOW STEP 1/3: PTP_J -> OVERHEAD OBSERVE POSE")
    print("========================================")
    node.send_joint_pose(
        observe_joints,
        velocity=joint_velocity,
        acc_time=acc_time,
    )
    node.wait_joint_arrived(observe_joints)
    node.settle(1.0)

    # Make the 3-view locator use the fixed taught overhead pose as its center.
    data = load_teach_config(config_path)
    data["apriltag_search_anchor_tool_pose"] = [float(v) for v in anchor]
    data["apriltag_search_anchor_saved_at_unix"] = float(time.time())
    save_teach_config(config_path, data)

    print("\n========================================")
    print("AUTO WORKFLOW STEP 2/3: THREE-VIEW APRILTAG LOCALIZATION")
    print("========================================")
    localize_three_views(
        node=node,
        config_path=config_path,
        T_G_C=T_G_C,
        offset=offset,
        velocity=velocity,
        acc_time=acc_time,
        samples_per_view=samples_per_view,
    )

    localized = load_teach_config(config_path)
    T_Base_Tag_raw = localized.get("T_Base_AprilTag")
    if T_Base_Tag_raw is None:
        raise RuntimeError("三視角定位結束後沒有 T_Base_AprilTag")
    T_Base_Tag = np.asarray(T_Base_Tag_raw, dtype=np.float64)

    print("\n========================================")
    print("AUTO WORKFLOW STEP 3/3: MOVE TO TAUGHT FINAL OBSERVATION")
    print("========================================")

    # Rebuild the same camera relation to the newly localized AprilTag.
    T_Base_CameraGoal = T_Base_Tag @ T_Tag_CameraFinal
    T_Base_ToolGoal = T_Base_CameraGoal @ invert_T(T_G_C)
    final_tool_pose = T_to_pose(T_Base_ToolGoal)

    print("new T_Base_AprilTag:")
    print(np.array2string(T_Base_Tag, precision=8, separator=", "))
    print("")
    print("reconstructed T_Base_CameraGoal:")
    print(np.array2string(
        T_Base_CameraGoal,
        precision=8,
        separator=", ",
    ))
    print("")
    print("TM final tool target [x,y,z,rx,ry,rz]:")
    print(np.array2string(
        np.asarray(final_tool_pose),
        precision=8,
        separator=", ",
    ))

    pre_adjust_joint2(
        node=node,
        offset_deg=joint2_pre_offset_deg,
        joint_velocity=joint_velocity,
        acc_time=acc_time,
    )

    print("Planning/executing final PTP_T from the adjusted Joint2 start state...")
    node.send_pose(
        final_tool_pose,
        velocity=velocity,
        acc_time=acc_time,
        motion="PTP_T",
    )
    node.wait_arrived(np.asarray(final_tool_pose, dtype=np.float64))
    node.settle(0.5)

    print("\n========================================")
    print("AUTO WORKFLOW COMPLETE")
    print("========================================")
    print("Overhead pose -> 3-view AprilTag -> taught final observation pose")
    print("========================================")



def measure_current_tag(
    node,
    config_path,
    T_G_C,
    samples_per_view,
):
    """Measure AprilTag once from the current robot pose.

    Uses the same convention as the user's calibration program:
        T_Base_Tag = T_Base_G @ T_G_C @ T_C_Tag

    The robot does NOT move in this mode.
    """
    node.spin_until_ready(
        need_camera=True,
        timeout=12.0,
    )

    actual_pose = [
        float(v)
        for v in node.current_tool_pose[:6]
    ]
    actual_joints = (
        [float(v) for v in node.current_joint_positions[:6]]
        if node.current_joint_positions is not None
        else None
    )
    actual_joint_names = (
        list(node.current_joint_names)
        if node.current_joint_names is not None
        else None
    )

    T_Base_G = pose_to_T(actual_pose)

    T_C_Tag = node.detect_tag_stable(
        count=max(1, int(samples_per_view)),
        timeout=6.0,
    )

    T_Base_Tag = (
        T_Base_G
        @ T_G_C
        @ T_C_Tag
    )

    T_Tag_Base = invert_T(T_Base_Tag)

    data = load_teach_config(config_path)
    data["latest_single_measure_tool_pose"] = actual_pose
    data["latest_single_measure_joint_names"] = actual_joint_names
    data["latest_single_measure_joint_positions_rad"] = actual_joints
    data["latest_single_T_Camera_AprilTag"] = T_C_Tag.tolist()
    data["latest_single_T_Base_AprilTag"] = T_Base_Tag.tolist()
    data["latest_single_T_AprilTag_Base"] = T_Tag_Base.tolist()
    data["latest_single_measure_at_unix"] = float(time.time())
    save_teach_config(config_path, data)

    print("\n========================================")
    print("APRILTAG CURRENT-POSE MEASUREMENT")
    print("========================================")
    print("Formula:")
    print("T_Base_Tag = T_Base_G @ T_G_C @ T_C_Tag")
    print("")
    print("Current TM tool_pose [x,y,z,rx,ry,rz]:")
    print(np.array2string(
        np.asarray(actual_pose),
        precision=8,
        separator=", ",
    ))
    print("")
    print("T_Base_G:")
    print(np.array2string(
        T_Base_G,
        precision=8,
        separator=", ",
    ))
    print("")
    print("T_G_C (Panel selected Eye-in-Hand matrix):")
    print(np.array2string(
        T_G_C,
        precision=8,
        separator=", ",
    ))
    print("")
    print("T_C_Tag (AprilTag detector):")
    print(np.array2string(
        T_C_Tag,
        precision=8,
        separator=", ",
    ))
    print("")
    print("T_Base_AprilTag:")
    print(np.array2string(
        T_Base_Tag,
        precision=8,
        separator=", ",
    ))
    print("")
    print(
        "AprilTag XYZ in Base [m] =",
        np.array2string(
            T_Base_Tag[:3, 3],
            precision=8,
            separator=", ",
        ),
    )
    print("")
    print("saved to:")
    print(config_path)
    print("========================================")


def show_config(config_path):
    data = load_teach_config(config_path)
    print("\n========================================")
    print("APRILTAG TEACH CONFIG")
    print("========================================")
    print("path:", config_path)

    anchor = data.get(
        "apriltag_search_anchor_tool_pose"
    )
    print("anchor:", anchor)

    T = data.get("T_Base_AprilTag")
    if T is not None:
        T = np.asarray(T, dtype=np.float64)
        print("T_Base_AprilTag:")
        print(np.array2string(
            T,
            precision=8,
            separator=", ",
        ))
        print(
            "Tag XYZ in Base:",
            np.array2string(
                T[:3, 3],
                precision=8,
                separator=", ",
            ),
        )
    else:
        print("T_Base_AprilTag: 尚未定位")

    print("")
    print("fixed observe pose file:", OBSERVE_POSE_FILE)
    print("exists:", OBSERVE_POSE_FILE.is_file())
    print("fixed final transform file:", FINAL_TRANSFORM_FILE)
    print("exists:", FINAL_TRANSFORM_FILE.is_file())
    print("three-view image dir:", THREE_VIEW_IMAGE_DIR)
    print("========================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "record",
            "measure",
            "localize",
            "record_final",
            "run_full",
            "step_check",
            "step_observe_joint",
            "step_final_joint",
            "step_preview_final",
            "step_move_final",
            "show",
        ],
    )

    parser.add_argument(
        "--eye-yaml",
        default="",
    )
    parser.add_argument(
        "--eye-key",
        default="T_G_C",
    )
    parser.add_argument(
        "--eye-invert",
        action="store_true",
    )

    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.130,
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--lateral-offset",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--joint-velocity",
        type=float,
        default=0.20,
        help="PTP_J joint velocity in rad/s",
    )
    parser.add_argument(
        "--joint2-pre-offset-deg",
        type=float,
        default=1.0,
        help=(
            "Before final PTP_T, add this signed degree offset to current Joint2. "
            "Positive means current J2 + offset."
        ),
    )
    parser.add_argument(
        "--acc-time",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--samples-per-view",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--image-topic",
        default=(
            "/eye_in_hand/"
            "eye_in_hand_camera/color/image_raw"
        ),
    )
    parser.add_argument(
        "--info-topic",
        default=(
            "/eye_in_hand/"
            "eye_in_hand_camera/color/camera_info"
        ),
    )
    parser.add_argument(
        "--no-return-center",
        action="store_true",
        help="完成三視角後不要回 CENTER；統整觀察流程使用。",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )

    args = parser.parse_args()
    config_path = Path(
        os.path.expanduser(args.config)
    ).resolve()

    if args.mode == "show":
        show_config(config_path)
        return

    rclpy.init()

    node = AprilTagTeachNode(
        tag_size=args.tag_size,
        tag_id=args.tag_id,
        image_topic=args.image_topic,
        info_topic=args.info_topic,
    )

    try:
        if not args.eye_yaml:
            raise RuntimeError(
                f"{args.mode} 模式需要 Eye-in-Hand YAML / T_G_C"
            )

        T_G_C = load_matrix(
            args.eye_yaml,
            args.eye_key,
            args.eye_invert,
        )

        if args.mode == "record":
            record_anchor(
                node,
                config_path,
                T_G_C,
            )
            return

        if args.mode == "record_final":
            record_final_pose(
                node,
                config_path,
                T_G_C,
            )
            return

        if args.mode == "run_full":
            run_full_workflow(
                node=node,
                config_path=config_path,
                T_G_C=T_G_C,
                offset=float(args.lateral_offset),
                velocity=float(args.velocity),
                joint_velocity=float(args.joint_velocity),
                joint2_pre_offset_deg=float(args.joint2_pre_offset_deg),
                acc_time=float(args.acc_time),
                samples_per_view=max(
                    1,
                    int(args.samples_per_view),
                ),
            )
            return

        if args.mode == "step_check":
            step_check_data(
                node,
                config_path,
                T_G_C,
            )
            return

        if args.mode == "step_observe_joint":
            step_move_observe_joint(
                node=node,
                joint_velocity=float(args.joint_velocity),
                acc_time=float(args.acc_time),
            )
            return

        if args.mode == "step_final_joint":
            step_move_recorded_final_joint(
                node=node,
                joint_velocity=float(args.joint_velocity),
                acc_time=float(args.acc_time),
            )
            return

        if args.mode == "step_preview_final":
            step_preview_final(
                node,
                config_path,
                T_G_C,
            )
            return

        if args.mode == "step_move_final":
            step_move_final(
                node=node,
                config_path=config_path,
                T_G_C=T_G_C,
                velocity=float(args.velocity),
                joint_velocity=float(args.joint_velocity),
                joint2_pre_offset_deg=float(args.joint2_pre_offset_deg),
                acc_time=float(args.acc_time),
            )
            return

        if args.mode == "measure":
            measure_current_tag(
                node=node,
                config_path=config_path,
                T_G_C=T_G_C,
                samples_per_view=max(
                    1,
                    int(args.samples_per_view),
                ),
            )
            return

        localize_three_views(
            node=node,
            config_path=config_path,
            T_G_C=T_G_C,
            offset=float(args.lateral_offset),
            velocity=float(args.velocity),
            acc_time=float(args.acc_time),
            samples_per_view=max(
                1,
                int(args.samples_per_view),
            ),
            return_to_center=not bool(args.no_return_center),
        )

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
