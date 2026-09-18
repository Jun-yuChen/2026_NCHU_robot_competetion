import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from dt_apriltags import Detector
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from tm_msgs.msg import FeedbackState
import yaml

# ==================== 驗證 TM 姿態角是「外旋 XYZ」還是「內旋 xyz」====================
#
# 原理：單一姿態沒辦法分辨內旋/外旋（只有一軸有角度時兩種慣例算出來一樣）。
# 但同一顆固定不動的 Tag，從兩個角度明顯不同的視角各量一次，算出來的
# T_Base_Tag 理論上該是同一個答案——慣例對，兩次結果幾乎一樣；慣例錯，
# 兩次結果會明顯對不上，且視角差越大、露出來的誤差越大。
#
# 這支腳本純粹「觀察」：只讀相機/法蘭回饋，完全不送任何移動指令，
# 全程由人手動 FreeDrive，不會有任何自動運動風險。
#
# 用法：
#   ros2 run py_gripper verify_euler_convention
# 可選參數：--convention xyz  （用內旋 xyz 重跑，比較兩種慣例的結果）


def pose_to_T(pose, convention="XYZ"):
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R.from_euler(convention, pose[3:6], degrees=False).as_matrix()
    T[0:3, 3] = pose[0:3]
    return T


def load_umi_hand_eye(config_path="ICA_Lab_UMI_Config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return np.array(cfg["T_G_C"], dtype=np.float64)


def rotation_angle_deg(R_a, R_b):
    """兩個旋轉矩陣之間的「角度差」（度），跟 Euler 角表示法無關，不會有換慣例失真的問題"""
    relative = R.from_matrix(R_a.T @ R_b)
    return np.degrees(relative.magnitude())


class EulerConventionVerifier(Node):

    def __init__(
        self,
        umi_config_path="ICA_Lab_UMI_Config.yaml",
        image_topic="/camera/camera/color/image_raw",
        info_topic="/camera/camera/color/camera_info",
        tag_id=0,
        tag_size=0.13,
    ):
        super().__init__("verify_euler_convention")

        self.T_Flange_Camera = load_umi_hand_eye(umi_config_path)
        self.TAG_ID = tag_id
        self.TAG_SIZE = tag_size

        self.cv_bridge = CvBridge()
        self.detector = Detector(
            searchpath=["apriltags"],
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        self.current_tool_pose = None
        self.camera_info = None
        self.latest_frame = None
        self.latest_color_frame = None

        self.preview_window_name = "Euler Convention Verify"
        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.preview_window_name, 848, 480)

        self.create_subscription(
            FeedbackState, "feedback_states", self._feedback_cb, 10
        )
        self.create_subscription(CameraInfo, info_topic, self._info_cb, 10)
        self.create_subscription(Image, image_topic, self._image_cb, 10)

    def _feedback_cb(self, msg):
        self.current_tool_pose = list(msg.tool_pose)

    def _info_cb(self, msg):
        self.camera_info = {
            "fx": msg.k[0],
            "fy": msg.k[4],
            "cx": msg.k[2],
            "cy": msg.k[5],
        }

    def _image_cb(self, msg):
        frame = self.cv_bridge.imgmsg_to_cv2(msg)
        self.latest_color_frame = frame
        self.latest_frame = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )

    def get_current_flange_pose(self):
        while self.current_tool_pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        return list(self.current_tool_pose)

    def _camera_params(self):
        if self.camera_info is None:
            return None
        return (
            self.camera_info["fx"],
            self.camera_info["fy"],
            self.camera_info["cx"],
            self.camera_info["cy"],
        )

    def _draw_overlay(self, results, selected):
        display = (
            self.latest_color_frame.copy()
            if self.latest_color_frame.ndim == 3
            else cv2.cvtColor(self.latest_color_frame, cv2.COLOR_GRAY2BGR)
        )
        for r in results:
            color = (0, 255, 0) if r is selected else (0, 0, 255)
            corners = r.corners.astype(int)
            for i in range(4):
                cv2.line(
                    display, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, 2
                )
        return display

    def detect_single_frame_tag(self):
        rclpy.spin_once(self, timeout_sec=0.1)
        frame = self.latest_frame
        params = self._camera_params()
        if frame is None or params is None:
            return None, [], None

        results = self.detector.detect(frame, True, params, self.TAG_SIZE)
        candidates = [r for r in results if r.tag_id == self.TAG_ID]
        selected = max(
            candidates,
            key=lambda r: np.linalg.norm(r.corners[0] - r.corners[2]),
            default=None,
        )
        if selected is None:
            return None, results, None

        T_c_tag = np.eye(4, dtype=np.float64)
        T_c_tag[:3, :3] = selected.pose_R
        T_c_tag[:3, 3] = selected.pose_t.reshape(3)
        return T_c_tag, results, selected

    def detect_tag_stable(self, samples=10):
        poses = []
        for _ in range(samples):
            T_c_tag, results, selected = self.detect_single_frame_tag()
            if T_c_tag is not None:
                poses.append(T_c_tag)
            display = self._draw_overlay(results, selected)
            cv2.imshow(self.preview_window_name, display)
            cv2.waitKey(1)
            time.sleep(0.03)

        if len(poses) < 3:
            return None

        translations = [p[:3, 3] for p in poses]
        median_t = np.median(translations, axis=0)
        mean_rot = R.from_matrix([p[:3, :3] for p in poses]).mean()

        T_stable = np.eye(4, dtype=np.float64)
        T_stable[:3, :3] = mean_rot.as_matrix()
        T_stable[:3, 3] = median_t
        return T_stable

    def _wait_confirm_and_capture(self, label, samples=10):
        print(f"\n[{label}] 請 FreeDrive 相機到指定角度，看得到 Tag 後點視窗按 Enter/空白鍵確認（Q/ESC 取消）...")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_frame is not None and self.camera_info is not None:
                _, results, selected = self.detect_single_frame_tag()
                display = self._draw_overlay(results, selected)
                cv2.imshow(self.preview_window_name, display)
            key = cv2.waitKey(15) & 0xFF
            if key in (13, 10, 32):
                break
            if key in (27, ord("q")):
                return None, None

        flange_pose = self.get_current_flange_pose()
        print(f"[{label}] 停穩取樣中 ({samples} 幀)...")
        time.sleep(0.3)
        T_camera_tag = self.detect_tag_stable(samples=samples)
        if T_camera_tag is None:
            print(f"[{label}] 沒有穩定偵測到 Tag，這次量測作廢。")
            return None, None
        return flange_pose, T_camera_tag

    def run(self, convention):
        print(f"\n===== 驗證慣例: {convention} =====")
        print(f"目標 Tag id={self.TAG_ID}, size={self.TAG_SIZE}m")

        flange_a, T_camera_tag_a = self._wait_confirm_and_capture("視角 A")
        if flange_a is None:
            print("已取消。")
            return

        flange_b, T_camera_tag_b = self._wait_confirm_and_capture("視角 B（請明顯換個角度，例如傾斜 30~40 度看同一顆 Tag）")
        if flange_b is None:
            print("已取消。")
            return

        T_Base_Tag_a = (
            pose_to_T(flange_a, convention) @ self.T_Flange_Camera @ T_camera_tag_a
        )
        T_Base_Tag_b = (
            pose_to_T(flange_b, convention) @ self.T_Flange_Camera @ T_camera_tag_b
        )

        pos_delta_mm = (
            np.linalg.norm(T_Base_Tag_a[:3, 3] - T_Base_Tag_b[:3, 3]) * 1000.0
        )
        rot_delta_deg = rotation_angle_deg(T_Base_Tag_a[:3, :3], T_Base_Tag_b[:3, :3])

        print(f"\n===== 結果（慣例: {convention}）=====")
        print(f"兩次視角量到的 T_Base_Tag 位置差: {pos_delta_mm:.1f} mm")
        print(f"兩次視角量到的 T_Base_Tag 旋轉差: {rot_delta_deg:.2f} deg")
        if pos_delta_mm < 15.0 and rot_delta_deg < 3.0:
            print("=> 差距在偵測雜訊範圍內，這個慣例大概率是對的。")
        else:
            print("=> 差距明顯偏大，這個慣例很可能是錯的，建議換另一個慣例重測比較。")


def main(args=None):
    rclpy.init(args=args)

    # 已用實測確認 TM 是內旋 xyz（外旋 45.3mm/16.6° vs 內旋 15.5mm/2.1°），預設用 xyz，
    # 想重新比較外旋可以加 --convention XYZ
    convention = "XYZ" if "--convention" in sys.argv and "XYZ" in sys.argv else "xyz"

    node = EulerConventionVerifier(umi_config_path="ICA_Lab_UMI_Config.yaml")
    node.run(convention)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
