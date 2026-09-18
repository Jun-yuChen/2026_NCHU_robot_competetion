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
    """[x, y, z, rx, ry, rz] (m, rad) -> 4x4 齊次矩陣 (TM 採內旋 xyz)"""
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
    T[0:3, 3] = pose[0:3]
    return T


def T_to_pose(T, ref_pose=None):
    """4x4 齊次矩陣 -> [x, y, z, rx, ry, rz] (m, rad)"""
    rot_mat = T[0:3, 0:3]
    trans = T[0:3, 3]
    e1 = np.array(R.from_matrix(rot_mat).as_euler("xyz", degrees=False))

    if ref_pose is None:
        return [
            float(trans[0]),
            float(trans[1]),
            float(trans[2]),
            float(e1[0]),
            float(e1[1]),
            float(e1[2]),
        ]

    # 計算對偶分支 e2
    e2 = np.zeros(3)
    e2[0] = e1[0] - np.pi if e1[0] > 0 else e1[0] + np.pi
    e2[1] = np.pi - e1[1]
    if e2[1] > np.pi:
        e2[1] -= 2 * np.pi
    elif e2[1] < -np.pi:
        e2[1] += 2 * np.pi
    e2[2] = e1[2] - np.pi if e1[2] > 0 else e1[2] + np.pi

    ref_rot = np.array(ref_pose[3:6])

    def unwrap_vec(angles, ref):
        res = np.copy(angles)
        for i in range(3):
            diff = res[i] - ref[i]
            while diff > np.pi:
                res[i] -= 2 * np.pi
                diff = res[i] - ref[i]
            while diff < -np.pi:
                res[i] += 2 * np.pi
                diff = res[i] - ref[i]
        return res

    e1_cand = unwrap_vec(e1, ref_rot)
    e2_cand = unwrap_vec(e2, ref_rot)

    best_e = (
        e1_cand
        if np.sum((e1_cand - ref_rot) ** 2) <= np.sum((e2_cand - ref_rot) ** 2)
        else e2_cand
    )

    return [
        float(trans[0]),
        float(trans[1]),
        float(trans[2]),
        float(best_e[0]),
        float(best_e[1]),
        float(best_e[2]),
    ]


def unwrap_euler(target_pose, current_pose):
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


class FindComputerTargetNew(Node):

    def __init__(
        self,
        umi_config_path="ICA_Lab_UMI_Config.yaml",
        image_topic="/camera/camera/color/image_raw",
        info_topic="/camera/camera/color/camera_info",
    ):
        super().__init__("find_computer_targetnew")

        self.T_Flange_Camera = load_umi_hand_eye(umi_config_path)

        self.TAG_ID = 0
        self.TAG_SIZE = 0.1375

        self.HOLE_NAMES = ["第 1 條線 (如電源線)", "第 2 條線 (如訊號線)"]

        self.PULL_DISTANCE_M = 0.12  # 拔線直線位移 12 cm
        self.EXTRACTION_AXIS = "Z"  # 沿夾爪中軸局部 Z 軸直線抽拔

        self.TRANSIT_VELOCITY, self.TRANSIT_ACC_TIME = 0.18, 0.2
        self.APPROACH_VELOCITY, self.APPROACH_ACC_TIME = 0.08, 0.3
        self.PULL_VELOCITY, self.PULL_ACC_TIME = 0.03, 0.4

        self.GRIPPER_OPEN = 0.0
        self.GRIPPER_CLOSE = 1.0

        self.SETTLE_TIME_S = 0.4

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

        self.preview_window_name = "Computer Target Navigator"
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

    def get_current_flange_pose(self):
        while self.current_tool_pose is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        return list(self.current_tool_pose)

    # ---------------- 運動與夾爪控制 ----------------

    def arm_move(
        self,
        pose,
        velocity=0.15,
        acc_time=0.2,
        motion_type=SetPositions.Request.PTP_T,
        error=0.002,
        rot_error_deg=0.5,
        timeout_sec=12.0,
    ):
        current = self.get_current_flange_pose()
        safe_pose = unwrap_euler(pose, current)

        # 1. 空間半徑安全檢查 (半徑放寬至 0.69m，避免正常作業誤攔截)
        radius = np.linalg.norm(safe_pose[0:3])
        if radius > 0.695:
            self.get_logger().error(
                f"【安全攔截】目標距離 {radius:.2f}m 超出 TM5-700 物理極限 (0.70m)！"
            )
            raise RuntimeError(f"目標距離過遠 ({radius:.2f}m)，已安全停機。")

        # 2. 正前方約束 (主機必定在前方 X > 0.15m)
        if safe_pose[0] < 0.15:
            self.get_logger().error(f"【座標反向攔截】目標 X={safe_pose[0]:.2f}m 跑到身後！")
            raise RuntimeError("目標座標異常，已安全停機。")

        pos_str = [round(v, 3) for v in safe_pose[0:3]]
        delta_pos = [round((safe_pose[i] - current[i]) * 1000, 1) for i in range(3)]
        print(f"[下發移動] 目標 xyz={pos_str}(m) | 增量 Δxyz={delta_pos}(mm)")

        req = SetPositions.Request()
        req.motion_type = motion_type
        req.positions = list(safe_pose)
        req.velocity = velocity
        req.acc_time = acc_time
        req.blend_percentage = 0
        req.fine_goal = True

        future = self.pos_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        res = future.result()
        if res is None or not res.ok:
            raise RuntimeError(f"【手臂拒絕執行指令】目標超限: {safe_pose}")

        rot_error_rad = np.radians(rot_error_deg)
        start = time.time()
        stable_hits = 0
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.03)
            self._show_preview()
            curr = self.current_tool_pose
            if curr is not None:
                dist_sq = sum((safe_pose[i] - curr[i]) ** 2 for i in range(3))
                rot_ok = all(
                    abs((safe_pose[i] - curr[i] + np.pi) % (2 * np.pi) - np.pi)
                    <= rot_error_rad
                    for i in range(3, 6)
                )
                if dist_sq <= error**2 and rot_ok:
                    stable_hits += 1
                    if stable_hits >= 2:
                        return
                else:
                    stable_hits = 0
        self.get_logger().warn("運動抵達判斷逾時，繼續往下執行")

    def send_gripper(self, state):
        req = SetIO.Request()
        req.module = 1
        req.type = 1
        req.pin = 0
        req.state = float(state)
        future = self.io_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        time.sleep(0.6)

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

    def detect_tag_stable(self, samples=25):
        """原地定點多幀採樣 (25 幀中位數濾波)"""
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
            time.sleep(0.025)

        if len(poses) < max(4, int(samples * 0.35)):
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
        if not self._wait_confirm(
            "\n[步驟 1/3] 請將手臂移至【桌上 Tag 俯視特寫點】(相機距 Tag 30~35cm、微傾斜)..."
        ):
            return
        tag_inspect_pose = self.get_current_flange_pose()

        print("消除微震中，開始採樣 25 幀基準...")
        time.sleep(self.SETTLE_TIME_S)
        T_Camera_Tag_ref = self.detect_tag_stable(samples=25)
        if T_Camera_Tag_ref is None:
            raise RuntimeError("特寫點未穩定辨識到 AprilTag，示教中止！")

        T_Base_Flange_inspect_ref = pose_to_T(tag_inspect_pose)
        T_Base_Tag_ref = (
            T_Base_Flange_inspect_ref
            @ self.T_Flange_Camera
            @ T_Camera_Tag_ref
        )
        inv_T_Base_Tag_ref = np.linalg.inv(T_Base_Tag_ref)
        T_Tag_Inspect = inv_T_Base_Tag_ref @ T_Base_Flange_inspect_ref
        print("桌面 AprilTag 空間原點基準鎖定成功。")

        T_Tag_Holes = []
        flange_holes_taught = []
        for i, name in enumerate(self.HOLE_NAMES, start=1):
            if not self._wait_confirm(
                f"\n[步驟 {1 + i}/3] 請 FreeDrive 將夾爪夾住【{name}】(到位夾緊姿態)..."
            ):
                return
            hole_flange_pose = self.get_current_flange_pose()
            flange_holes_taught.append(list(hole_flange_pose))
            T_Base_HoleFlange = pose_to_T(hole_flange_pose)
            T_Tag_Hole = inv_T_Base_Tag_ref @ T_Base_HoleFlange
            T_Tag_Holes.append(T_Tag_Hole.tolist())
            print(f"{name} 相對 Tag 幾何矩陣已儲存。")

        config_data = {
            "tag_inspect_pose": [float(v) for v in tag_inspect_pose],
            "T_Base_Tag_ref": T_Base_Tag_ref.tolist(),
            "T_Tag_Inspect": T_Tag_Inspect.tolist(),
            "T_Tag_Holes": T_Tag_Holes,
            "flange_holes_taught": flange_holes_taught,
            "hole_names": self.HOLE_NAMES,
        }
        with open(save_path, "w") as f:
            yaml.safe_dump(config_data, f)
        print(f"\n示教完成！幾何設定檔已寫入 {save_path}")

    # ==================== 流程二：自動拔線 (SE(2) 示教姿態剛性繼承) ====================

    def run_unplugging(self, config_path="cable_unplug_config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        tag_inspect_pose = cfg.get("tag_inspect_pose") or cfg.get(
            "high_vantage_pose"
        )
        T_Tag_Inspect = (
            np.array(cfg["T_Tag_Inspect"], dtype=np.float64)
            if "T_Tag_Inspect" in cfg
            else None
        )
        T_Tag_Holes = [np.array(m, dtype=np.float64) for m in cfg["T_Tag_Holes"]]
        hole_names = cfg.get("hole_names", self.HOLE_NAMES)

        # 1. 重構基準標籤姿態 T_Base_Tag_ref
        if "T_Base_Tag_ref" in cfg and cfg["T_Base_Tag_ref"] is not None:
            T_Base_Tag_ref = np.array(cfg["T_Base_Tag_ref"], dtype=np.float64)
        else:
            T_Base_Flange_insp = pose_to_T(tag_inspect_pose)
            T_Base_Tag_ref = (
                T_Base_Flange_insp @ np.linalg.inv(T_Tag_Inspect)
                if T_Tag_Inspect is not None
                else T_Base_Flange_insp
            )

        # 2. 重構各孔位的原始示教姿態 (Flange 6D pose)
        flange_holes_taught = []
        if "flange_holes_taught" in cfg:
            flange_holes_taught = cfg["flange_holes_taught"]
        else:
            for T_th in T_Tag_Holes:
                T_bh = T_Base_Tag_ref @ T_th
                flange_holes_taught.append(T_to_pose(T_bh))

        # ---------------- 階段一：平穩移至特寫點 (安全高度只抬 8 公分) ----------------
        high_transit_pose = [
            tag_inspect_pose[0],
            tag_inspect_pose[1],
            tag_inspect_pose[2] + 0.08,  # 嚴格限制高度，絕不撞破 70cm 臂展
            tag_inspect_pose[3],
            tag_inspect_pose[4],
            tag_inspect_pose[5],
        ]

        print("\n[階段 1/2] 移至特寫點上方平穩過渡點...")
        self.arm_move(
            high_transit_pose,
            velocity=self.TRANSIT_VELOCITY,
            motion_type=SetPositions.Request.PTP_T,
        )

        print("[階段 1/2] 垂直下降進入桌上 Tag 俯視特寫點...")
        self.arm_move(
            tag_inspect_pose,
            velocity=self.APPROACH_VELOCITY,
            motion_type=SetPositions.Request.PTP_T,
        )

        # ---------------- 階段二：原地鎖死 25 幀高精度時域採樣 ----------------
        print(f"\n[階段 2/2] 原地停穩消震 ({self.SETTLE_TIME_S}s)...")
        time.sleep(self.SETTLE_TIME_S)

        print("[階段 2/2] 正在原地採樣 25 幀解算 (手臂完全鎖死不動)...")
        T_Camera_Tag = self.detect_tag_stable(samples=25)
        if T_Camera_Tag is None:
            raise RuntimeError("特寫點未偵測到 Tag！請確認標籤是否反光。")

        flange_actual = self.get_current_flange_pose()
        T_Base_Tag_curr = (
            pose_to_T(flange_actual) @ self.T_Flange_Camera @ T_Camera_Tag
        )

        # ---------------- 核心演算法：計算推車相對於桌子的 (ΔX, ΔY, ΔYaw) ----------------
        tag_pos_ref = T_Base_Tag_ref[:3, 3]
        tag_pos_curr = T_Base_Tag_curr[:3, 3]

        R_ref = T_Base_Tag_ref[:3, :3]
        R_curr = T_Base_Tag_curr[:3, :3]
        R_rel = R_ref.T @ R_curr

        # 提取地面水平旋轉角 Yaw
        delta_yaw = np.arctan2(R_rel[1, 0], R_rel[0, 0])

        # 消除 180° 翻轉噪聲
        if abs(delta_yaw) > np.radians(45.0):
            delta_yaw = (
                delta_yaw - np.pi if delta_yaw > 0 else delta_yaw + np.pi
            )

        # 物理硬體限幅：推車偏角絕對在 ±15° 內
        delta_yaw = float(
            np.clip(delta_yaw, np.radians(-15.0), np.radians(15.0))
        )

        # 計算平移差 (ΔX, ΔY, ΔZ)
        delta_pos = tag_pos_curr - tag_pos_ref
        print(
            f"[空間補償] 推車停靠偏差：ΔX={delta_pos[0]*1000:.1f}mm, "
            f"ΔY={delta_pos[1]*1000:.1f}mm, ΔYaw={np.degrees(delta_yaw):.2f}°"
        )

        # ---------------- 階段三：依序拔除兩條線材 (剛性繼承示教姿態) ----------------
        for i, (hole_taught, name) in enumerate(
            zip(flange_holes_taught, hole_names), start=1
        ):
            print(f"\n{'=' * 20} 執行拔除第 {i}/2 條線：{name} {'=' * 20}")

            # 1. 空間幾何平移旋轉補償 (圍繞 Tag 原點做 2D 剛體旋轉)
            dx = hole_taught[0] - tag_pos_ref[0]
            dy = hole_taught[1] - tag_pos_ref[1]

            grasp_x = tag_pos_curr[0] + (
                dx * np.cos(delta_yaw) - dy * np.sin(delta_yaw)
            )
            grasp_y = tag_pos_curr[1] + (
                dx * np.sin(delta_yaw) + dy * np.cos(delta_yaw)
            )
            grasp_z = hole_taught[2] + delta_pos[2]

            # 2. 姿態角剛性鎖定：Rx, Ry 100% 維持示教角度，Rz 僅補償微小偏航角
            # 徹底杜絕 180° 翻轉引發底座轉向後方 (Scorpion Pose)
            grasp_rx = hole_taught[3]
            grasp_ry = hole_taught[4]
            grasp_rz = hole_taught[5] + delta_yaw

            grasp_pose = [
                grasp_x,
                grasp_y,
                grasp_z,
                grasp_rx,
                grasp_ry,
                grasp_rz,
            ]

            # 3. 沿夾爪中軸 (Tool Z 軸) 向外推算退避點 prep_pose
            R_grasp = R.from_euler(
                "xyz", [grasp_rx, grasp_ry, grasp_rz], degrees=False
            ).as_matrix()
            z_tool = R_grasp[:, 2]  # 工具坐標系的 Z 軸在 Base 下的方向

            prep_pos = np.array([grasp_x, grasp_y, grasp_z]) - (
                self.PULL_DISTANCE_M * z_tool
            )
            prep_pose = [
                float(prep_pos[0]),
                float(prep_pos[1]),
                float(prep_pos[2]),
                grasp_rx,
                grasp_ry,
                grasp_rz,
            ]

            # 4. 高空過渡點：只在預備點垂直上方抬高 8 公分 (作業半徑嚴格 < 0.62m)
            high_prep_pose = [
                prep_pose[0],
                prep_pose[1],
                prep_pose[2] + 0.08,
                grasp_rx,
                grasp_ry,
                grasp_rz,
            ]

            print(f"[動作 1/6] 平穩移至 {name} 軸線上方過渡點...")
            self.arm_move(
                high_prep_pose,
                velocity=self.TRANSIT_VELOCITY,
                motion_type=SetPositions.Request.PTP_T,
            )
            self.send_gripper(self.GRIPPER_OPEN)

            print(f"[動作 2/6] 垂直下切至進場預備點...")
            self.arm_move(
                prep_pose,
                velocity=self.APPROACH_VELOCITY,
                motion_type=SetPositions.Request.PTP_T,
                error=0.002,
            )

            print(f"[動作 3/6] LINE_T 直線切入咬合插頭...")
            self.arm_move(
                grasp_pose,
                velocity=self.APPROACH_VELOCITY,
                motion_type=SetPositions.Request.LINE_T,
                error=0.0015,
                rot_error_deg=0.3,
            )
            self.send_gripper(self.GRIPPER_CLOSE)

            print(
                f"[動作 4/6] LINE_T 筆直抽拔線材 (拔出 {self.PULL_DISTANCE_M * 100:.1f} cm)..."
            )
            self.arm_move(
                prep_pose,
                velocity=self.PULL_VELOCITY,
                motion_type=SetPositions.Request.LINE_T,
            )

            print(f"[動作 5/6] 開爪釋放線頭...")
            self.send_gripper(self.GRIPPER_OPEN)

            print(f"[動作 6/6] 垂直抬升至安全高度...")
            self.arm_move(
                high_prep_pose,
                velocity=self.TRANSIT_VELOCITY,
                motion_type=SetPositions.Request.PTP_T,
            )

        print("\n2 條線材已全數拔除完成！")


# ==================== 程式執行入口 ====================
def main(args=None):
    rclpy.init(args=args)
    unplugger = FindComputerTargetNew(
        umi_config_path="ICA_Lab_UMI_Config.yaml"
    )

    if "--teach" in sys.argv:
        unplugger.run_teaching(save_path="cable_unplug_config.yaml")
    else:
        unplugger.run_unplugging(config_path="cable_unplug_config.yaml")

    unplugger.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()