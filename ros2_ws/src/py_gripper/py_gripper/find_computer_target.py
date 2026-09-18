import sys
import time
import cv2
from cv_bridge import CvBridge
from dt_apriltags import Detector
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SetIO, SetPositions
import yaml

# ==================== 1. 座標轉換與幾何工具 ====================


def pose_to_T(pose):
    """[x, y, z, rx, ry, rz] (m, rad) -> 4x4 齊次矩陣

    TM 實測是內旋 xyz，不是外旋 XYZ——用 verify_euler_convention.py
    實測驗證過（外旋誤差 45.3mm/16.6°，內旋誤差降到 15.5mm/2.1°）。
    """
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
    T[0:3, 3] = pose[0:3]
    return T


def T_to_pose(T):
    """4x4 齊次矩陣 -> [x, y, z, rx, ry, rz] (m, rad)"""
    rot = R.from_matrix(T[0:3, 0:3]).as_euler("xyz", degrees=False)
    trans = T[0:3, 3]
    return [
        float(trans[0]),
        float(trans[1]),
        float(trans[2]),
        float(rot[0]),
        float(rot[1]),
        float(rot[2]),
    ]


def unwrap_euler(target_pose, current_pose):
    """將目標角度對齊當前角度，消除 +/- 180 度 (+/- pi) 的旋轉跳躍"""
    if current_pose is None:
        return list(target_pose)
    adjusted = list(target_pose)
    for i in range(3, 6):
        diff = adjusted[i] - current_pose[i]
        while diff > np.pi:
            adjusted[i] -= 2 * np.pi
            diff = adjusted[i] - current_pose[i]
        while diff < -np.pi:
            adjusted[i] += 2 * np.pi
            diff = adjusted[i] - current_pose[i]
    return adjusted


def load_umi_hand_eye(config_path="ICA_Lab_UMI_Config.yaml"):
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到手眼標定檔 {config_path}") from None
    return np.array(cfg["T_G_C"], dtype=np.float64)


# ==================== 2. 車載高精度插拔控制類別 ====================


class DynamicCableUnplugger(Node):

    def __init__(
        self,
        umi_config_path="ICA_Lab_UMI_Config.yaml",
        image_topic="/camera/camera/color/image_raw",
        info_topic="/camera/camera/color/camera_info",
    ):
        super().__init__("dynamic_cable_unplugger")

        # 載入手眼轉換矩陣
        self.T_Flange_Camera = load_umi_hand_eye(umi_config_path)

        # 桌面固定 AprilTag 設定 (實體印刷尺寸: 0.1375m，實測值)
        self.TAG_ID = 0
        self.TAG_SIZE = 0.1375

        # 待拔線材定義 (依序執行 2 條線)
        self.HOLE_NAMES = ["第 1 條線 (如電源線)", "第 2 條線 (如訊號線)"]

        # 拔線行程與抽拔軸向設定
        self.PULL_DISTANCE_M = 0.12  # 拔線直線位移 12 cm
        self.EXTRACTION_AXIS = "Z"  # 沿夾爪工具坐標系 Z 軸向外筆直抽拔

        # 各階段運動速度與加速度 (m/s, s)
        self.TRANSIT_VELOCITY, self.TRANSIT_ACC_TIME = 0.18, 0.2
        self.APPROACH_VELOCITY, self.APPROACH_ACC_TIME = 0.08, 0.3
        self.PULL_VELOCITY, self.PULL_ACC_TIME = 0.03, 0.4

        # 夾爪狀態 (0.0: 開爪, 1.0: 合爪)
        self.GRIPPER_OPEN = 0.0
        self.GRIPPER_CLOSE = 1.0

        # 手臂到位消震延遲時間 (秒)
        self.SETTLE_TIME_S = 0.5

        self.cv_bridge = CvBridge()
        self.detector = Detector(
            searchpath=["apriltags"],
            families="tag36h11",
            nthreads=2,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        self.current_tool_pose = None
        self.camera_info = None
        self.camera_matrix = None
        self.camera_distortion = None
        self.latest_frame = None
        self.latest_color_frame = None

        self.preview_window_name = "Dynamic Tag Navigator Preview"
        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.preview_window_name, 848, 480)

        self.create_subscription(
            FeedbackState, "feedback_states", self._feedback_cb, 10
        )
        self.create_subscription(CameraInfo, info_topic, self._info_cb, 10)
        self.create_subscription(Image, image_topic, self._image_cb, 10)

        self.pos_cli = self.create_client(SetPositions, "set_positions")
        while not self.pos_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("等待手臂 /set_positions 服務上線...")

        self.io_cli = self.create_client(SetIO, "/set_io")
        while not self.io_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("等待夾爪 /set_io 服務上線...")

    # ---------------- ROS2 通訊回呼 ----------------

    def _feedback_cb(self, msg):
        self.current_tool_pose = list(msg.tool_pose)

    def _info_cb(self, msg):
        self.camera_info = {
            "fx": msg.k[0],
            "fy": msg.k[4],
            "cx": msg.k[2],
            "cy": msg.k[5],
        }
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.camera_distortion = np.array(msg.d, dtype=np.float64)

    def _image_cb(self, msg):
        frame = self.cv_bridge.imgmsg_to_cv2(msg)
        self.latest_color_frame = frame
        self.latest_frame = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.ndim == 3
            else frame
        )

    # ---------------- 運動與夾爪控制 ----------------

    def get_current_flange_pose(self):
        while self.current_tool_pose is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        return list(self.current_tool_pose)

    def arm_move(
        self,
        pose,
        velocity=0.15,
        acc_time=0.2,
        motion_type=SetPositions.Request.PTP_T,
        error=0.005,
        rot_error_deg=1.0,
        timeout_sec=15.0,
    ):
        """高精度抵達運動指令 (error 5mm、角度 1 度；原本 1mm/0.3 度太緊，硬體回饋雜訊常達不到，導致每次都逾時)"""
        current = self.get_current_flange_pose()
        safe_pose = unwrap_euler(pose, current)

        req = SetPositions.Request()
        req.motion_type = motion_type
        req.positions = list(safe_pose)
        req.velocity = velocity
        req.acc_time = acc_time
        req.blend_percentage = 0
        req.fine_goal = True

        future = self.pos_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if not future.result().ok:
            raise RuntimeError(f"手臂拒絕了移動指令: {safe_pose}")

        rot_error_rad = np.radians(rot_error_deg)
        start = time.time()
        stable_hits = 0
        stable_hits_required = 3  # 連續 3 次快照都在容差內才算真的到位，過濾穩定抖動雜訊
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.03)
            self._show_preview()
            current = self.current_tool_pose
            if current is not None:
                dist_sq = sum(
                    (safe_pose[i] - current[i]) ** 2 for i in range(3)
                )
                rot_ok = all(
                    abs(
                        (safe_pose[i] - current[i] + np.pi) % (2 * np.pi)
                        - np.pi
                    )
                    <= rot_error_rad
                    for i in range(3, 6)
                )
                if dist_sq <= error**2 and rot_ok:
                    stable_hits += 1
                    if stable_hits >= stable_hits_required:
                        return
                else:
                    stable_hits = 0
        current = self.current_tool_pose
        if current is not None:
            residual_mm = [round((safe_pose[i] - current[i]) * 1000, 1) for i in range(3)]
            residual_deg = [
                round(np.degrees((safe_pose[i] - current[i] + np.pi) % (2 * np.pi) - np.pi), 2)
                for i in range(3, 6)
            ]
            self.get_logger().warn(
                f"運動抵達判斷逾時，繼續往下執行 "
                f"(殘餘 Δxyz={residual_mm}mm, Δrpy={residual_deg}deg)"
            )
        else:
            self.get_logger().warn("運動抵達判斷逾時，繼續往下執行（未收到法蘭回饋）")

    def send_gripper(self, state):
        req = SetIO.Request()
        req.module = 1
        req.type = 1
        req.pin = 0
        req.state = float(state)
        future = self.io_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        time.sleep(0.6)

    def compute_tool_offset_pose(self, base_pose, offset_distance):
        """沿工具坐標系自身軸向進行筆直前進或後退 (右乘)"""
        T_Base_Tool = pose_to_T(base_pose)
        T_Tool_Offset = np.eye(4, dtype=np.float64)

        if self.EXTRACTION_AXIS == "Z":
            T_Tool_Offset[2, 3] = offset_distance
        elif self.EXTRACTION_AXIS == "X":
            T_Tool_Offset[0, 3] = offset_distance

        return T_to_pose(T_Base_Tool @ T_Tool_Offset)

    # ---------------- 視覺處理：去畸變、濾波與擺頭巡邏 ----------------

    def _show_preview(self, overlay_text=None):
        if self.latest_color_frame is None:
            return
        display = self.latest_color_frame.copy()
        if overlay_text:
            cv2.putText(
                display,
                overlay_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        cv2.imshow(self.preview_window_name, display)
        cv2.waitKey(1)

    def detect_tag_stable(self, samples=15):
        """去畸變並進行 15 幀四元數與平移中位數濾波"""
        while self.camera_info is None or self.latest_frame is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        poses = []
        for _ in range(samples):
            rclpy.spin_once(self, timeout_sec=0.03)
            undistorted = cv2.undistort(
                self.latest_frame, self.camera_matrix, self.camera_distortion
            )
            params = (
                self.camera_info["fx"],
                self.camera_info["fy"],
                self.camera_info["cx"],
                self.camera_info["cy"],
            )
            results = self.detector.detect(
                undistorted, True, params, self.TAG_SIZE
            )

            matches = [r for r in results if r.tag_id == self.TAG_ID]
            if matches:
                selected = max(
                    matches,
                    key=lambda r: np.linalg.norm(r.corners[0] - r.corners[2]),
                )
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = selected.pose_R
                T[:3, 3] = selected.pose_t.reshape(3)
                poses.append(T)
            time.sleep(0.035)

        if len(poses) < max(3, int(samples * 0.4)):
            return None

        translations = np.array([p[:3, 3] for p in poses])
        median_t = np.median(translations, axis=0)
        mean_rot = (
            R.from_matrix([p[:3, :3] for p in poses]).mean().as_matrix()
        )

        T_stable = np.eye(4, dtype=np.float64)
        T_stable[:3, :3] = mean_rot
        T_stable[:3, 3] = median_t
        return T_stable

    def find_tag_with_patrol(self, samples=5):
        """原地鎖定 XYZ，轉動手腕微視角搜尋標籤 (避開光影死角)"""
        patrol_angles = [
            (0.0, 0.0),  # 1. 原角度正視
            (-12.0, 0.0),  # 2. 向左轉 12 度
            (12.0, 0.0),  # 3. 向右轉 12 度
            (0.0, -8.0),  # 4. 向上微抬 8 度
            (0.0, 8.0),  # 5. 向下微壓 8 度
        ]

        flange_start = self.get_current_flange_pose()
        T_Base_Start = pose_to_T(flange_start)
        R_Flange_Camera = self.T_Flange_Camera[0:3, 0:3]
        inv_R_Flange_Camera = R_Flange_Camera.T

        for pan_deg, tilt_deg in patrol_angles:
            if pan_deg != 0.0 or tilt_deg != 0.0:
                r_cam_local = R.from_euler(
                    "xyz",
                    [np.radians(tilt_deg), np.radians(pan_deg), 0.0],
                    degrees=False,
                ).as_matrix()
                R_flange_delta = (
                    R_Flange_Camera @ r_cam_local @ inv_R_Flange_Camera
                )

                T_target = np.eye(4, dtype=np.float64)
                T_target[0:3, 3] = T_Base_Start[0:3, 3]  # 鎖死空間位置
                T_target[0:3, 0:3] = T_Base_Start[0:3, 0:3] @ R_flange_delta

                self.arm_move(
                    T_to_pose(T_target),
                    velocity=0.15,
                    motion_type=SetPositions.Request.PTP_T,
                )
                time.sleep(0.3)

            tag_pose = self.detect_tag_stable(samples=samples)
            if tag_pose is not None:
                return tag_pose

        return None

    def _wait_confirm(self, prompt):
        print(prompt)
        print("(請點選預覽視窗按 Enter 確認；按 Q 取消)")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            self._show_preview("Align Tag / Confirm")
            key = cv2.waitKey(15) & 0xFF
            if key in (13, 10, 32):
                return True
            if key in (27, ord("q")):
                return False

    # ==================== 流程一：示教模式 ====================

    def run_teaching(self, save_path="cable_unplug_config.yaml"):
        # 步驟 1: 示教高空廣角搜尋姿態 (距桌面約 60~70cm，視野涵蓋大半張桌子)
        if not self._wait_confirm(
            "\n[步驟 1/4] 請將手臂移至【高空廣角搜尋點】(距桌面約 60~70cm，確保看到桌上 Tag)..."
        ):
            return
        high_vantage_pose = self.get_current_flange_pose()

        # 步驟 2: 示教最佳俯視特寫點 (相機距 Tag 約 30~35cm、微傾斜 10~15 度、夾爪指尖距桌面 >10cm)
        if not self._wait_confirm(
            "\n[步驟 2/4] 請將手臂移至【桌上 Tag 俯視特寫點】(相機距 Tag 30~35cm、微傾斜避開頂燈反光)..."
        ):
            return
        tag_inspect_pose = self.get_current_flange_pose()

        print("消除微震中，開始採樣 15 幀基準...")
        time.sleep(self.SETTLE_TIME_S)
        T_Camera_Tag_ref = self.detect_tag_stable(samples=15)
        if T_Camera_Tag_ref is None:
            raise RuntimeError("特寫點未穩定辨識到 AprilTag，示教中止！")

        T_Base_Flange_inspect_ref = pose_to_T(tag_inspect_pose)
        T_Base_Tag_ref = (
            T_Base_Flange_inspect_ref
            @ self.T_Flange_Camera
            @ T_Camera_Tag_ref
        )
        inv_T_Base_Tag_ref = np.linalg.inv(T_Base_Tag_ref)

        # 算得特寫點相對於 Tag 的相對變換矩陣
        T_Tag_Inspect = inv_T_Base_Tag_ref @ T_Base_Flange_inspect_ref
        print("桌面 AprilTag 參考基準與動態特寫幾何常數鎖定成功。")

        # 步驟 3 & 4: 依序示教 2 個線材插頭的到位夾取姿態
        T_Tag_Holes = []
        for i, name in enumerate(self.HOLE_NAMES, start=1):
            if not self._wait_confirm(
                f"\n[步驟 {2 + i}/4] 請 FreeDrive 將夾爪夾住【{name}】(完全到位夾緊姿態)..."
            ):
                return
            hole_flange_pose = self.get_current_flange_pose()
            T_Base_HoleFlange = pose_to_T(hole_flange_pose)

            # 解耦相對位姿常數：T_Tag_Hole = (T_Base_Tag_ref)^-1 * T_Base_HoleFlange
            T_Tag_Hole = inv_T_Base_Tag_ref @ T_Base_HoleFlange
            T_Tag_Holes.append(T_Tag_Hole.tolist())
            print(f"{name} 相對 Tag 幾何矩陣已計算儲存。")

        config_data = {
            "high_vantage_pose": [float(v) for v in high_vantage_pose],
            "T_Tag_Inspect": T_Tag_Inspect.tolist(),
            "T_Tag_Holes": T_Tag_Holes,
            "hole_names": self.HOLE_NAMES,
        }
        with open(save_path, "w") as f:
            yaml.safe_dump(config_data, f)
        print(f"\n示教完成！幾何設定檔已寫入 {save_path}")

    # ==================== 流程二：動態下潛與雙孔位拔線 ====================

    def run_unplugging(self, config_path="cable_unplug_config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        high_vantage_pose = cfg["high_vantage_pose"]
        T_Tag_Inspect = np.array(cfg["T_Tag_Inspect"], dtype=np.float64)
        T_Tag_Holes = [np.array(m, dtype=np.float64) for m in cfg["T_Tag_Holes"]]
        hole_names = cfg.get("hole_names", self.HOLE_NAMES)

        # ---------------- 階段一：高空大視野捕捉 ----------------
        print("\n[階段 1/3] 移至高空廣角搜尋點 (視野涵蓋大半張桌面)...")
        self.arm_move(
            high_vantage_pose,
            velocity=self.TRANSIT_VELOCITY,
            motion_type=SetPositions.Request.PTP_T,
        )
        time.sleep(0.4)

        print("[階段 1/3] 正在高空搜尋桌上 AprilTag...")
        T_Camera_Tag_rough = self.find_tag_with_patrol(samples=5)
        if T_Camera_Tag_rough is None:
            raise RuntimeError(
                "安全中斷：高空視野仍找不到 AprilTag！請確認推車停靠位置偏差是否過大。"
            )

        flange_high = self.get_current_flange_pose()
        T_Base_Tag_rough = (
            pose_to_T(flange_high) @ self.T_Flange_Camera @ T_Camera_Tag_rough
        )
        print("高空粗鎖定成功！")

        # ---------------- 階段二：動態計算特寫點並精準下潛 ----------------
        # 關鍵：特寫點由當前粗抓的 Tag 座標動態乘出，不受推車停靠誤差影響
        T_Base_Inspect_dynamic = T_Base_Tag_rough @ T_Tag_Inspect
        inspect_pose = T_to_pose(T_Base_Inspect_dynamic)

        # 門型軌跡：高空平移至正上方後再垂直下降
        safe_z = max(flange_high[2], inspect_pose[2] + 0.15, 0.45)
        transit_pose = [
            inspect_pose[0],
            inspect_pose[1],
            safe_z,
            inspect_pose[3],
            inspect_pose[4],
            inspect_pose[5],
        ]

        print("\n[階段 2/3] 高空水平移至動態特寫點正上方...")
        self.arm_move(
            transit_pose,
            velocity=self.TRANSIT_VELOCITY,
            motion_type=SetPositions.Request.PTP_T,
        )

        print("[階段 2/3] 垂直切入動態俯視特寫點...")
        self.arm_move(
            inspect_pose,
            velocity=self.APPROACH_VELOCITY,
            motion_type=SetPositions.Request.PTP_T,
        )

        # ---------------- 階段三：停穩 15 幀高精度二次採樣 ----------------
        print(f"[階段 3/3] 消除微震中 ({self.SETTLE_TIME_S}s)...")
        time.sleep(self.SETTLE_TIME_S)

        print("[階段 3/3] 啟動 15 幀次毫米級標籤位姿解算...")
        T_Camera_Tag_fine = self.detect_tag_stable(samples=15)
        if T_Camera_Tag_fine is None:
            print("[警告] 特寫點單張未抓到，啟動特寫微幅擺頭搜尋...")
            T_Camera_Tag_fine = self.find_tag_with_patrol(samples=8)

        if T_Camera_Tag_fine is None:
            raise RuntimeError("特寫點二次定位失敗，請確認標籤表面是否有嚴重反光！")

        flange_inspect_actual = self.get_current_flange_pose()
        T_Base_Tag_final = (
            pose_to_T(flange_inspect_actual)
            @ self.T_Flange_Camera
            @ T_Camera_Tag_fine
        )
        print("高精度 AprilTag 原點建立完成，準備開始拔線動作。")

        # ---------------- 階段四：依序執行 2 條線材的拔除 ----------------
        for i, (T_Tag_Hole, name) in enumerate(
            zip(T_Tag_Holes, hole_names), start=1
        ):
            print(f"\n{'=' * 20} 執行拔除第 {i}/2 條線：{name} {'=' * 20}")

            # 動態還原孔位目標
            T_Base_Grasp = T_Base_Tag_final @ T_Tag_Hole
            grasp_pose = T_to_pose(T_Base_Grasp)

            # 沿夾爪工具軸向局部退避計算預備點
            prep_pose = self.compute_tool_offset_pose(
                grasp_pose, -self.PULL_DISTANCE_M
            )

            # 高空過渡點 (避開主機機殼與角鋼)
            hole_safe_z = max(safe_z, prep_pose[2] + 0.15)
            high_prep_pose = [
                prep_pose[0],
                prep_pose[1],
                hole_safe_z,
                prep_pose[3],
                prep_pose[4],
                prep_pose[5],
            ]

            # 動作 1: 高空移至孔位外側上方
            print(f"[動作 1/6] 高空移至 {name} 軸線正上方...")
            self.arm_move(
                high_prep_pose,
                velocity=self.TRANSIT_VELOCITY,
                motion_type=SetPositions.Request.PTP_T,
            )
            self.send_gripper(self.GRIPPER_OPEN)

            # 動作 2: 垂直下降至進場預備點
            print(f"[動作 2/6] 垂直下降至進場預備點...")
            self.arm_move(
                prep_pose,
                velocity=self.APPROACH_VELOCITY,
                motion_type=SetPositions.Request.PTP_T,
            )

            # 動作 3: LINE_T 直線切入咬合插頭
            print(f"[動作 3/6] LINE_T 直線切入咬合插頭...")
            self.arm_move(
                grasp_pose,
                velocity=self.APPROACH_VELOCITY,
                motion_type=SetPositions.Request.LINE_T,
            )
            self.send_gripper(self.GRIPPER_CLOSE)

            # 動作 4: LINE_T 筆直抽拔線材 (Unplug)
            print(
                f"[動作 4/6] LINE_T 筆直抽拔線材 (拔出 {self.PULL_DISTANCE_M * 100:.1f} cm)..."
            )
            self.arm_move(
                prep_pose,
                velocity=self.PULL_VELOCITY,
                motion_type=SetPositions.Request.LINE_T,
            )

            # 動作 5: 開爪釋放已拔出線材
            print(f"[動作 5/6] 開爪釋放線頭...")
            self.send_gripper(self.GRIPPER_OPEN)

            # 動作 6: 垂直抬升高空，準備下一個動作
            print(f"[動作 6/6] 垂直抬升至安全高空...")
            self.arm_move(
                high_prep_pose,
                velocity=self.TRANSIT_VELOCITY,
                motion_type=SetPositions.Request.PTP_T,
            )

        print("\n2 條線材已全數拔除完成！")


# ==================== 程式執行入口 ====================
def main(args=None):
    rclpy.init(args=args)
    unplugger = DynamicCableUnplugger(
        umi_config_path="ICA_Lab_UMI_Config.yaml"
    )

    # 示教模式：ros2 run py_gripper dynamic_cable_unplugger --teach
    if "--teach" in sys.argv:
        unplugger.run_teaching(save_path="cable_unplug_config.yaml")
    else:
        unplugger.run_unplugging(config_path="cable_unplug_config.yaml")

    unplugger.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()