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

# ==================== 1. 座標轉換與工具函式 ====================


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
    """將 target_pose 的角度對齊 current_pose，消除 +/- 180 度 (+/- pi) 的旋轉跳躍"""
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
    """從 UMI 設定檔中讀取 T_Flange_Camera (即 T_G_C)"""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"找不到手眼標定檔 {config_path}，"
            "請確認執行指令時的工作目錄，或改傳完整路徑。"
        ) from None
    T_Flange_Camera = np.array(cfg["T_G_C"], dtype=np.float64)
    return T_Flange_Camera


# ==================== 2. 機械手臂與視覺整合類別 ====================


class BigToSmallTagNavigator(Node):

    def __init__(
        self,
        umi_config_path="ICA_Lab_UMI_Config.yaml",
        image_topic="/camera/camera/color/image_raw",
        info_topic="/camera/camera/color/camera_info",
    ):
        super().__init__("tag_navigator")

        # 載入手眼矩陣 T_G_C
        self.T_Flange_Camera = load_umi_hand_eye(umi_config_path)

        # 標籤實體尺寸 (單位: 公尺，實測值)
        # 大小 Tag 目前印刷內容相同、id 都是 0，無法靠 id 分辨，
        # 偵測時一律靠像素跨距挑最大/最小的那個來區分（見 _pixel_span / pick 參數）。
        self.BIG_TAG_ID = 0
        self.BIG_TAG_SIZE = 0.13  # 13 cm

        self.SMALL_TAG_ID = 0
        self.SMALL_TAG_SIZE = 0.0475  # 4.75 cm

        # 【手腕掃描視角】: 鎖定 XYZ 不動，僅旋轉手腕 (Pan 左右, Tilt 上下)，單位: 度
        self.PATROL_ANGLES_DEG = [
            (0.0, 0.0),  # 1. 示教原角度正視
            (-12.0, 0.0),  # 2. 鏡頭向左轉 12 度
            (12.0, 0.0),  # 3. 鏡頭向右轉 12 度
            (0.0, -8.0),  # 4. 鏡頭微抬 8 度
            (0.0, 8.0),  # 5. 鏡頭下壓 8 度
        ]

        # 各階段移動速度/加速時間 (m/s, s)：越靠近目標、動作越精細的階段用越慢的速度
        self.HOME_VELOCITY, self.HOME_ACC_TIME = 0.2, 0.2
        self.SCAN_VELOCITY, self.SCAN_ACC_TIME = 0.15, 0.2
        self.TRANSIT_VELOCITY, self.TRANSIT_ACC_TIME = 0.2, 0.2
        self.FINAL_APPROACH_VELOCITY, self.FINAL_APPROACH_ACC_TIME = 0.12, 0.3

        # 抵達觀察點後，重試偵測小 Tag 的最長等待時間 (s)
        self.SMALL_TAG_DETECT_TIMEOUT_S = 8.0

        # 夾爪 IO 狀態，比照 arm_goto_fixed_table.py 的 module=1/type=1/pin=0 慣例
        self.GRIPPER_OPEN_STATE = 0.0
        self.GRIPPER_CLOSE_STATE = 1.0

        # 抓取點的預備點：沿 Base 座標系 -X 方向往後退這個距離 (m)，比照 arm_goto_fixed_table.py
        self.GRASP_RETREAT_M = 0.15

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

        # 預覽視窗
        self.preview_window_name = "Tag Navigator Preview"
        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.preview_window_name, 848, 480)
        placeholder = np.zeros((480, 848, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Waiting for camera...",
            (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow(self.preview_window_name, placeholder)
        cv2.waitKey(1)

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

    # ---------------- ROS2 訂閱回呼 ----------------

    def _feedback_cb(self, msg):
        self.current_tool_pose = list(msg.tool_pose)

    def _info_cb(self, msg):
        self.camera_info = {
            "fx": msg.k[0],
            "fy": msg.k[4],
            "cx": msg.k[2],
            "cy": msg.k[5],
        }

    def _camera_params(self):
        """回傳 (fx, fy, cx, cy)，camera_info 還沒到就回 None"""
        if self.camera_info is None:
            return None
        return (
            self.camera_info["fx"],
            self.camera_info["fy"],
            self.camera_info["cx"],
            self.camera_info["cy"],
        )

    def _image_cb(self, msg):
        frame = self.cv_bridge.imgmsg_to_cv2(msg)
        self.latest_color_frame = frame
        self.latest_frame = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.ndim == 3
            else frame
        )

    # ---------------- 底層運動控制 ----------------

    def get_current_flange_pose(self):
        """讀取當前法蘭在 Base 下的 6 軸卡氏座標 [x, y, z, rx, ry, rz]"""
        while self.current_tool_pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        return list(self.current_tool_pose)

    def arm_move_ptp(
        self,
        pose,
        velocity=0.2,
        acc_time=0.2,
        blend_percentage=0,
        fine_goal=True,
        error=0.005,
        rot_error_deg=1.0,
        timeout_sec=15.0,
        motion_type=None,
    ):
        """驅動 TM 手臂移動至目標卡氏座標，並進行角度防跳轉保護

        motion_type 預設 PTP_T；短距離精準移動（如夾取點前後這種）
        建議傳 SetPositions.Request.LINE_T，比照 arm_goto_fixed_table.py 的慣例。
        """
        if motion_type is None:
            motion_type = SetPositions.Request.PTP_T
        current = self.get_current_flange_pose()
        safe_pose = unwrap_euler(pose, current)

        pos_str = [round(v, 3) for v in safe_pose[0:3]]
        rot_deg_str = [round(np.degrees(v), 1) for v in safe_pose[3:6]]
        delta_pos = [round(safe_pose[i] - current[i], 3) for i in range(3)]
        delta_rot = [
            round(np.degrees(safe_pose[i] - current[i]), 1) for i in range(3, 6)
        ]

        print(
            f"\n[移動目標] xyz={pos_str} (m), rpy={rot_deg_str} (deg)\n"
            f"[位姿增量] Δxyz={delta_pos} (m), Δrpy={delta_rot} (deg)"
        )

        req = SetPositions.Request()
        req.motion_type = motion_type
        req.positions = list(safe_pose)
        req.velocity = velocity
        req.acc_time = acc_time
        req.blend_percentage = blend_percentage
        req.fine_goal = fine_goal

        future = self.pos_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is None or not result.ok:
            raise RuntimeError(
                f"arm_move_ptp: /set_positions 拒絕了移動指令 (result={result})，"
                "請確認手臂處於 Auto/Run 模式且目標姿態未超出關節極限。"
            )

        rot_error_rad = np.radians(rot_error_deg)
        start = time.time()
        stable_hits = 0
        stable_hits_required = 3  # 連續 3 次快照都在容差內才算真的到位，過濾穩定抖動雜訊
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            self._show_debug_frame()
            current = self.current_tool_pose
            if current is not None:
                dist_sq = sum((safe_pose[i] - current[i]) ** 2 for i in range(3))
                rot_ok = all(
                    abs((safe_pose[i] - current[i] + np.pi) % (2 * np.pi) - np.pi)
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
                f"arm_move_ptp: 抵達判斷逾時，繼續往下執行 "
                f"(殘餘 Δxyz={residual_mm}mm, Δrpy={residual_deg}deg)"
            )
        else:
            self.get_logger().warn("arm_move_ptp: 抵達判斷逾時，繼續往下執行（未收到法蘭回饋）")

    def _compute_prep_pose(self, grasp_pose):
        """抓取點沿 Base 座標系 -X 方向後退 GRASP_RETREAT_M，姿態角不變（比照 arm_goto_fixed_table.py）"""
        prep_pose = list(grasp_pose)
        prep_pose[0] -= self.GRASP_RETREAT_M
        return prep_pose

    def send_gripper(self, state):
        """控制夾爪張合，state=GRIPPER_OPEN_STATE(0.0) 開、GRIPPER_CLOSE_STATE(1.0) 合

        比照 arm_goto_fixed_table.py 的 set_gripper：走 TM 控制器的 /set_io，
        module=1/type=1/pin=0，不依賴 robotiq_85_msgs。
        """
        req = SetIO.Request()
        req.module = 1
        req.type = 1
        req.pin = 0
        req.state = float(state)

        future = self.io_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if not future.done() or future.result() is None:
            self.get_logger().error("send_gripper: /set_io 指令逾時，未收到服務回應！")

    # ---------------- 影像處理與標籤解算 ----------------

    @staticmethod
    def _pixel_span(result):
        c = result.corners
        d1 = np.linalg.norm(c[0] - c[2])
        d2 = np.linalg.norm(c[1] - c[3])
        return (d1 + d2) / 2.0

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
                    display,
                    tuple(corners[i]),
                    tuple(corners[(i + 1) % 4]),
                    color,
                    2,
                )
            center = tuple(r.center.astype(int))
            cv2.circle(display, center, 4, color, -1)
            cv2.putText(
                display,
                f"id={r.tag_id}",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return display

    def _show_debug_frame(self, results=None, selected=None):
        if self.latest_color_frame is None:
            return
        display = self._draw_overlay(results or [], selected)
        cv2.imshow(self.preview_window_name, display)
        cv2.waitKey(1)

    def _hold_preview(self, message):
        print(f"\n{message} (點選預覽視窗後按 Enter/空白鍵/Q/ESC 結束)")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            results = []
            params = self._camera_params()
            if self.latest_frame is not None and params is not None:
                # estimate_tag_pose=False：只需要框線/id 做預覽，不需要 tag_size
                results = self.detector.detect(self.latest_frame, False, params, None)
            self._show_debug_frame(results)
            key = cv2.waitKey(15) & 0xFF
            if key in (13, 10, 32, 27, ord("q")):
                break

    def detect_single_frame_tag(self, target_id, tag_size, pick="largest"):
        while self.camera_info is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        rclpy.spin_once(self, timeout_sec=0.1)
        frame = self.latest_frame
        if frame is None:
            return None

        results = self.detector.detect(frame, True, self._camera_params(), tag_size)
        candidates = [r for r in results if r.tag_id == target_id]
        selected = None
        if candidates:
            selected = (
                max(candidates, key=self._pixel_span)
                if pick == "largest"
                else min(candidates, key=self._pixel_span)
            )
        self._show_debug_frame(results, selected)

        if selected is None:
            return None
        T_c_tag = np.eye(4, dtype=np.float64)
        T_c_tag[:3, :3] = selected.pose_R
        T_c_tag[:3, 3] = selected.pose_t.reshape(3)
        return T_c_tag

    def detect_tag_stable(self, target_id, tag_size, samples=5, pick="largest"):
        poses = []
        for _ in range(samples):
            T_c_tag = self.detect_single_frame_tag(
                target_id, tag_size, pick=pick
            )
            if T_c_tag is not None:
                poses.append(T_c_tag)
            time.sleep(0.03)

        if len(poses) < 3:
            return None

        translations = [p[:3, 3] for p in poses]
        median_t = np.median(translations, axis=0)

        # 旋轉角也一起濾波，不能只濾平移：單幀雜訊會被原封不動放大到遠處目標
        mean_rotation = R.from_matrix([p[:3, :3] for p in poses]).mean()

        T_stable = np.eye(4, dtype=np.float64)
        T_stable[:3, :3] = mean_rotation.as_matrix()
        T_stable[:3, 3] = median_t
        return T_stable

    def _wait_for_confirm_with_preview(
        self, prompt, target_id, tag_size, pick="largest"
    ):
        print(prompt)
        print(
            "(請點一下預覽視窗讓它取得焦點，確認 Tag 框線對正後按 Enter 或空白鍵；按 Q 或 ESC 取消)"
        )

        confirmed = False
        start_time = time.time()
        warned_no_frame = False
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.latest_frame is None:
                if not warned_no_frame and time.time() - start_time > 3.0:
                    print(
                        "[警告] 3 秒內沒收到任何影像，請檢查相機節點與 Topic 名稱。"
                    )
                    warned_no_frame = True
                key = cv2.waitKey(15) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            results = []
            selected = None
            params = self._camera_params()
            if params is not None:
                results = self.detector.detect(
                    self.latest_frame, True, params, tag_size
                )
                candidates = [r for r in results if r.tag_id == target_id]
                if candidates:
                    selected = (
                        max(candidates, key=self._pixel_span)
                        if pick == "largest"
                        else min(candidates, key=self._pixel_span)
                    )

            self._show_debug_frame(results, selected)
            key = cv2.waitKey(15) & 0xFF
            if key in (13, 10, 32):
                confirmed = True
                break
            if key in (27, ord("q")):
                break

        return confirmed

    # ==================== 流程一：示教並萃取轉移矩陣 ====================

    def run_teaching(self, save_path="tag_nav_config.yaml"):
        if not self._wait_for_confirm_with_preview(
            "\n[示教步驟 1] 請將手臂移至【開闊俯瞰點】，避開夾爪遮擋，確認看到大 Tag 後確認...",
            self.BIG_TAG_ID,
            self.BIG_TAG_SIZE,
            pick="largest",
        ):
            print("[取消] 示教已取消。")
            return
        home_pose = self.get_current_flange_pose()

        time.sleep(0.3)
        T_Camera_BigTag = self.detect_tag_stable(
            self.BIG_TAG_ID, self.BIG_TAG_SIZE, samples=5, pick="largest"
        )
        if T_Camera_BigTag is None:
            print("[錯誤] 開闊點未穩定辨識到大 Tag，示教中止！")
            return

        T_Base_Flange1 = pose_to_T(home_pose)
        T_Base_BigTag = T_Base_Flange1 @ self.T_Flange_Camera @ T_Camera_BigTag
        inv_T_Base_BigTag = np.linalg.inv(T_Base_BigTag)
        print("大 Tag 全域基座位置鎖定成功。")

        if not self._wait_for_confirm_with_preview(
            "\n[示教步驟 2] 請 FreeDrive 將相機移至【小 Tag 正前方特寫約 8~12cm 處】，手肘請往上提維持 Elbow-Up，居中後確認...",
            self.SMALL_TAG_ID,
            self.SMALL_TAG_SIZE,
            pick="smallest",
        ):
            print("[取消] 示教已取消。")
            return
        inspect_pose = self.get_current_flange_pose()
        T_Base_InspectFlange = pose_to_T(inspect_pose)

        # 核心算式：T_BigTag_Inspect = (T_Base_BigTag)^-1 * T_Base_InspectFlange
        T_BigTag_Inspect = inv_T_Base_BigTag @ T_Base_InspectFlange

        # 在 inspect 點順便偵測一次小 Tag，當作步驟 3 的參考基準
        time.sleep(0.3)
        T_Camera_SmallTag_ref = self.detect_tag_stable(
            self.SMALL_TAG_ID, self.SMALL_TAG_SIZE, samples=5, pick="smallest"
        )
        if T_Camera_SmallTag_ref is None:
            print("[錯誤] inspect 點未穩定辨識到小 Tag，示教中止！")
            return
        T_Base_SmallTag_ref = (
            T_Base_InspectFlange @ self.T_Flange_Camera @ T_Camera_SmallTag_ref
        )
        inv_T_Base_SmallTag_ref = np.linalg.inv(T_Base_SmallTag_ref)

        if not self._wait_for_confirm_with_preview(
            "\n[示教步驟 3] 請 FreeDrive 將相機移至【小 Tag 最終貼近觀察點】(正對、貼近)，確認後確認...",
            self.SMALL_TAG_ID,
            self.SMALL_TAG_SIZE,
            pick="largest",
        ):
            print("[取消] 示教已取消。")
            return
        final_pose = self.get_current_flange_pose()
        T_Base_FinalFlange = pose_to_T(final_pose)

        # 核心算式：T_SmallTag_FinalPoint = (T_Base_SmallTag_ref)^-1 * T_Base_FinalFlange
        T_SmallTag_FinalPoint = inv_T_Base_SmallTag_ref @ T_Base_FinalFlange

        # 步驟 4~6：示教 3 個夾取點，一樣相對步驟 2 量到的小 Tag 算固定關係，
        # 不需要在這幾個點重新偵測小 Tag（夾取點鏡頭不一定看得到 Tag）
        T_SmallTag_GraspPoints = []
        for i in range(1, 4):
            if not self._wait_for_confirm_with_preview(
                f"\n[示教步驟 {3 + i}] 請 FreeDrive 將夾爪移至【夾取點 {i}】，確認後確認...",
                self.SMALL_TAG_ID,
                self.SMALL_TAG_SIZE,
                pick="largest",
            ):
                print("[取消] 示教已取消。")
                return
            grasp_pose = self.get_current_flange_pose()
            T_Base_GraspFlange = pose_to_T(grasp_pose)
            T_SmallTag_GraspPoints.append(
                (inv_T_Base_SmallTag_ref @ T_Base_GraspFlange).tolist()
            )

        config_data = {
            "home_vantage_pose": [float(v) for v in home_pose],
            "T_BigTag_Inspect": T_BigTag_Inspect.tolist(),
            "T_SmallTag_FinalPoint": T_SmallTag_FinalPoint.tolist(),
            "T_SmallTag_GraspPoints": T_SmallTag_GraspPoints,
        }
        with open(save_path, "w") as f:
            yaml.safe_dump(config_data, f)

        print(f"\n示教成功！相對變換矩陣已保存至 {save_path}")
        self._hold_preview("示教流程結束")

    # ==================== 流程二：自動導航至小 Tag 視野 ====================

    def run_navigation(self, config_path="tag_nav_config.yaml"):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"找不到 {config_path}，請先執行一次示教"
                "（ros2 run py_gripper tag_navigator --teach）產生這個檔案。"
            ) from None

        if "T_SmallTag_FinalPoint" not in cfg or "T_SmallTag_GraspPoints" not in cfg:
            raise KeyError(
                f"{config_path} 缺少 T_SmallTag_FinalPoint 或 T_SmallTag_GraspPoints，"
                "是舊版示教檔，請重新執行一次示教"
                "（ros2 run py_gripper tag_navigator --teach）。"
            )

        home_pose = cfg["home_vantage_pose"]
        T_BigTag_Inspect = np.array(cfg["T_BigTag_Inspect"], dtype=np.float64)
        T_SmallTag_FinalPoint = np.array(
            cfg["T_SmallTag_FinalPoint"], dtype=np.float64
        )
        T_SmallTag_GraspPoints = [
            np.array(m, dtype=np.float64) for m in cfg["T_SmallTag_GraspPoints"]
        ]

        # 1. 先平滑回到示教開闊基準點
        print("\n[執行] 移動至開闊基準點...")
        self.arm_move_ptp(
            home_pose, velocity=self.HOME_VELOCITY, acc_time=self.HOME_ACC_TIME
        )
        time.sleep(0.3)

        T_Base_Flange_home = pose_to_T(home_pose)
        R_Flange_Camera = self.T_Flange_Camera[0:3, 0:3]
        inv_R_Flange_Camera = R_Flange_Camera.T

        # 2. 僅轉動手腕進行相機 Pan/Tilt 視角搜尋
        T_Camera_BigTag_now = None
        for i, (pan_deg, tilt_deg) in enumerate(self.PATROL_ANGLES_DEG):
            print(
                f"\n[執行] 手腕掃描視角 {i + 1}/{len(self.PATROL_ANGLES_DEG)}: Pan={pan_deg}°, Tilt={tilt_deg}°"
            )

            # 相機光學座標系下的微旋轉矩陣 (繞 Y 左右 Pan, 繞 X 上下 Tilt)
            # 目前每筆只有單軸非零，內旋/外旋在這裡算出來一樣，改成 xyz 只是跟全檔案慣例一致
            r_cam_local = R.from_euler(
                "xyz",
                [np.radians(tilt_deg), np.radians(pan_deg), 0.0],
                degrees=False,
            ).as_matrix()

            # 轉換至法蘭座標系微旋轉: R_flange_rot = R_F_C * R_cam * R_C_F
            R_flange_delta = R_Flange_Camera @ r_cam_local @ inv_R_Flange_Camera

            # 計算目標姿態: 鎖死 XYZ 不動，只改變姿態角
            T_target_flange = np.eye(4, dtype=np.float64)
            T_target_flange[0:3, 3] = T_Base_Flange_home[0:3, 3]
            T_target_flange[0:3, 0:3] = (
                T_Base_Flange_home[0:3, 0:3] @ R_flange_delta
            )

            scan_waypoint = T_to_pose(T_target_flange)
            self.arm_move_ptp(
                scan_waypoint,
                velocity=self.SCAN_VELOCITY,
                acc_time=self.SCAN_ACC_TIME,
            )
            time.sleep(0.35)

            T_Camera_BigTag_now = self.detect_tag_stable(
                self.BIG_TAG_ID, self.BIG_TAG_SIZE, samples=5, pick="largest"
            )
            if T_Camera_BigTag_now is not None:
                print(f"[執行] 在視角 {i + 1} 成功鎖定大 Tag！")
                break

        if T_Camera_BigTag_now is None:
            raise RuntimeError(
                "安全中斷：手腕巡邏所有角度均未發現大 Tag，請確認標籤無反光或遮蔽！"
            )

        # 3. 讀取當下法蘭並透過矩陣鏈解算目標
        flange_now = self.get_current_flange_pose()
        T_Base_Flange_now = pose_to_T(flange_now)

        T_Base_BigTag_now = (
            T_Base_Flange_now @ self.T_Flange_Camera @ T_Camera_BigTag_now
        )
        T_Base_Target_Inspect = T_Base_BigTag_now @ T_BigTag_Inspect
        target_pose = T_to_pose(T_Base_Target_Inspect)

        # 4. 【門型安全軌跡移動】（防止手肘下沉撞擊桌面）
        print(f"\n[執行] 啟動門型安全路徑飛往小 Tag 觀察點...")

        # 步驟 4-1: 高空水平轉場點
        # 嚴格鎖定 Z 軸高度在開闊點高空 (或至少 0.45m 以上)，手肘絕不下墜
        safe_z = max(flange_now[2], target_pose[2] + 0.15, 0.45)
        transit_pose = [
            target_pose[0],  # 目標 X
            target_pose[1],  # 目標 Y
            safe_z,          # 維持高空安全 Z
            flange_now[3],   # 保持開闊點姿態 Rx
            flange_now[4],   # 保持開闊點姿態 Ry
            flange_now[5],   # 保持開闊點姿態 Rz
        ]

        print(
            f"[路徑 1/2] 高空水平移至目標正上方: X={round(transit_pose[0], 3)}, Y={round(transit_pose[1], 3)}, Z={round(safe_z, 3)}"
        )
        self.arm_move_ptp(
            transit_pose,
            velocity=self.TRANSIT_VELOCITY,
            acc_time=self.TRANSIT_ACC_TIME,
        )

        # 步驟 4-2: 在目標正上方就位後，再直線/關節下降至小 Tag 觀察點
        print(
            f"[路徑 2/2] 垂直切入小 Tag 觀察點: {[round(v, 3) for v in target_pose[:3]]}"
        )
        self.arm_move_ptp(
            target_pose,
            velocity=self.FINAL_APPROACH_VELOCITY,
            acc_time=self.FINAL_APPROACH_ACC_TIME,
        )

        print("已成功抵達小 Tag 視野範圍！")

        # 5. 即時偵測小 Tag，用示教好的 T_SmallTag_FinalPoint 算出最終貼近觀察點
        time.sleep(0.3)
        T_Camera_SmallTag_now = None
        retry_deadline = time.time() + self.SMALL_TAG_DETECT_TIMEOUT_S
        while time.time() < retry_deadline:
            T_Camera_SmallTag_now = self.detect_tag_stable(
                self.SMALL_TAG_ID, self.SMALL_TAG_SIZE, samples=5, pick="smallest"
            )
            if T_Camera_SmallTag_now is not None:
                break
        if T_Camera_SmallTag_now is None:
            raise RuntimeError(
                f"安全中斷：抵達觀察點後 {self.SMALL_TAG_DETECT_TIMEOUT_S:.0f} 秒內"
                "都沒偵測到小 Tag，無法精準定位！（若是被夾爪擋到邊緣，重試沒有用，"
                "需要重新示教留更多構圖 margin）"
            )

        flange_at_inspect = self.get_current_flange_pose()
        T_Base_Flange_inspect = pose_to_T(flange_at_inspect)
        T_Base_SmallTag_now = (
            T_Base_Flange_inspect @ self.T_Flange_Camera @ T_Camera_SmallTag_now
        )

        T_Base_Flange_final = T_Base_SmallTag_now @ T_SmallTag_FinalPoint
        final_target_pose = T_to_pose(T_Base_Flange_final)

        print("\n[執行] 移動到小 Tag 最終貼近觀察點...")
        self.arm_move_ptp(
            final_target_pose,
            velocity=self.FINAL_APPROACH_VELOCITY,
            acc_time=self.FINAL_APPROACH_ACC_TIME,
        )
        print("已抵達小 Tag 最終觀察點！")

        # 6. 依序執行 3 個夾取點：預備點開爪等待 → 抓取點合爪等待 → 退回預備點
        for i, T_SmallTag_Grasp in enumerate(T_SmallTag_GraspPoints, start=1):
            T_Base_Flange_grasp = T_Base_SmallTag_now @ T_SmallTag_Grasp
            grasp_pose = T_to_pose(T_Base_Flange_grasp)
            prep_pose = self._compute_prep_pose(grasp_pose)

            print(f"\n[執行] 夾取點 {i}/{len(T_SmallTag_GraspPoints)}：移動到預備點...")
            self.arm_move_ptp(
                prep_pose,
                velocity=self.TRANSIT_VELOCITY,
                acc_time=self.TRANSIT_ACC_TIME,
                motion_type=SetPositions.Request.PTP_T,
            )
            self.send_gripper(self.GRIPPER_OPEN_STATE)
            time.sleep(1.0)

            print(f"[執行] 夾取點 {i}/{len(T_SmallTag_GraspPoints)}：移動到抓取點...")
            self.arm_move_ptp(
                grasp_pose,
                velocity=self.FINAL_APPROACH_VELOCITY,
                acc_time=self.FINAL_APPROACH_ACC_TIME,
                motion_type=SetPositions.Request.LINE_T,
            )
            self.send_gripper(self.GRIPPER_CLOSE_STATE)
            time.sleep(1.0)

            print(f"[執行] 夾取點 {i}/{len(T_SmallTag_GraspPoints)}：退回預備點...")
            self.arm_move_ptp(
                prep_pose,
                velocity=self.FINAL_APPROACH_VELOCITY,
                acc_time=self.FINAL_APPROACH_ACC_TIME,
                motion_type=SetPositions.Request.LINE_T,
            )

        print("已完成 3 個夾取點！")
        self._hold_preview("導航流程結束")


# ==================== 主程式執行入口 ====================
def main(args=None):
    rclpy.init(args=args)

    navigator = BigToSmallTagNavigator(
        umi_config_path="ICA_Lab_UMI_Config.yaml"
    )

    # 用 --teach 切換示教模式，不加就是正式自動導航
    # 例: ros2 run py_gripper tag_navigator --teach
    if "--teach" in sys.argv:
        navigator.run_teaching(save_path="tag_nav_config.yaml")
    else:
        navigator.run_navigation(config_path="tag_nav_config.yaml")

    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
