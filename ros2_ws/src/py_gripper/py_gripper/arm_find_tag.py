import rclpy
from rclpy.node import Node
import time

import cv2
import numpy as np
from cv_bridge import CvBridge
from pupil_apriltags import Detector

from sensor_msgs.msg import Image
from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SetPositions

CAMERA_K = np.array([
    [615.0,   0.0, 424.0],
    [  0.0, 615.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

TAG_SIZE = 0.05                # 5 公分標籤
FINAL_HOVER_Z_M = 0.30         # 最終懸停高度 30cm
PATROL_SEARCH_Z_M = 0.40       # 高空巡邏高度 40cm

# 【90度垂直朝向桌面姿態基準】
# Rz: 141.12 deg -> 2.4630 rad 鎖死底座航向
STRICT_90DEG_DOWNWARD_RPY = [3.1415926, 0.0, 2.4630]

# 【相機像素軸 -> 底座 X/Y 軸的旋轉角，需要實機校正！】
# 目前 -90 度只是延用舊版寫死的 delta_x=+yc, delta_y=-xc，並非量測值。
# 校正方法：手臂固定在 STRICT_90DEG_DOWNWARD_RPY 姿態不動，沿底座 +X 點動一段
# 已知距離，觀察同一顆標籤在畫面上的 dx/dy 怎麼變，反推正確角度後填回這裡。
CAM_TO_BASE_YAW_RAD = -np.pi / 2.0

# 巡邏航點 (m)
SEARCH_WAYPOINTS_M = [
    [0.32,  0.00, PATROL_SEARCH_Z_M],
    [0.32,  0.12, PATROL_SEARCH_Z_M],
    [0.32, -0.12, PATROL_SEARCH_Z_M],
    [0.42,  0.00, PATROL_SEARCH_Z_M],
]

# 【STAGE_3 局部再搜尋】：抵達記憶點後看不到標籤，就地小範圍找一輪，
# 找不到才放棄記憶點、回到大範圍巡邏重新開始，避免卡在 RE-ACQUIRING 不動。
REACQUIRE_GRACE_S    = 1.2     # 看不到標籤超過這麼久，才開始局部搜尋
REACQUIRE_STEP_S     = 1.0     # 每個局部搜尋點的等待時間
REACQUIRE_SEARCH_Z_M = 0.38    # 局部搜尋時拉高一點，擴大視野
REACQUIRE_OFFSETS_M = [
    (0.00,  0.00),
    (0.05,  0.00), (-0.05,  0.00),
    (0.00,  0.05), ( 0.00, -0.05),
    (0.05,  0.05), (-0.05, -0.05),
    (0.05, -0.05), (-0.05,  0.05),
]

STAGE_0_WARMUP         = 0     # 開機靜止預熱 (防抽動)
STAGE_1_SEARCH         = 1     # 搜尋巡邏 (看見標籤立刻記憶)
STAGE_2_MEMORY_DASH    = 2     # 【直撲記憶點正上方 30cm】
STAGE_3_FINE_SERVO     = 3     # 【二次重捕獲・精確閉環微調】
STAGE_4_LOCKED         = 4     # 永久鎖死


class ArmMemoryGuidedAligner(Node):
    def __init__(self):
        super().__init__('arm_memory_guided_aligner')
        self.bridge = CvBridge()
        self.detector = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0)
        self.fx, self.fy = CAMERA_K[0, 0], CAMERA_K[1, 1]
        self.cx, self.cy = CAMERA_K[0, 2], CAMERA_K[1, 2]

        self.start_node_time = time.time()
        self.current_flange_pose_m = None
        self.feedback_count = 0
        self.stage = STAGE_0_WARMUP
        self.display_frame = None

        self.waypoint_idx = 0
        self.patrol_wait_start = time.time()
        self.patrol_moving = False

        self.memorized_target_hover_m = None
        self.dash_start_time = 0.0
        self.last_step_time = 0.0
        self.is_stepping = False
        self.bullseye_lock_count = 0

        self.lost_since_time = None
        self.reacquire_idx = 0
        self.reacquire_moving = False
        self.reacquire_move_time = 0.0

        self.win_name = "RealSense Memory-Guided Align"
        self.sub_feedback = self.create_subscription(FeedbackState, 'feedback_states', self.feedback_callback, 10)
        
        # 直接訂閱預設的 /camera/camera/color/image_raw，免除手動 remap
        self.sub_image = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 1)
        self.cli_set_pos = self.create_client(SetPositions, 'set_positions')

        self.get_logger().info("等待手臂 /set_positions 服務上線...")
        while not self.cli_set_pos.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("尚未連上 TM Robot 服務...")
        self.get_logger().info("【精準二次閉環微調版・正式修復版】已就緒！")

    def feedback_callback(self, msg: FeedbackState):
        raw_pose = msg.tool_pose
        if len(raw_pose) >= 6:
            p0, p1, p2 = raw_pose[0], raw_pose[1], raw_pose[2]
            if abs(p0) > 5.0 or abs(p1) > 5.0 or abs(p2) > 5.0:
                pos_m = [float(p0 / 1000.0), float(p1 / 1000.0), float(p2 / 1000.0)]
            else:
                pos_m = [float(p0), float(p1), float(p2)]

            rpy = [
                float(np.deg2rad(raw_pose[3])) if abs(raw_pose[3]) > 7.0 else float(raw_pose[3]),
                float(np.deg2rad(raw_pose[4])) if abs(raw_pose[4]) > 7.0 else float(raw_pose[4]),
                float(np.deg2rad(raw_pose[5])) if abs(raw_pose[5]) > 7.0 else float(raw_pose[5])
            ]

            self.current_flange_pose_m = pos_m + rpy
            self.feedback_count += 1

    def get_tag_analytic(self, tag):
        c = tag.corners
        s_px = (np.linalg.norm(c[1] - c[0]) +
                np.linalg.norm(c[2] - c[1]) +
                np.linalg.norm(c[3] - c[2]) +
                np.linalg.norm(c[0] - c[3])) / 4.0
        if s_px < 10.0:
            return None
        return (self.fx * TAG_SIZE) / s_px

    def cam_to_base_delta(self, dx_px, dy_px, dist_z):
        """
        將像素座標平面上的偏移，依 CAM_TO_BASE_YAW_RAD 旋轉映射到底座 X/Y。
        CAM_TO_BASE_YAW_RAD 目前設為 -90 度，等價於原本寫死的
        delta_x_base=+yc, delta_y_base=-xc，數值上完全相同。
        這個角度理論上該等於相機像素軸相對底座 X/Y 軸的實際夾角，
        需要用實機點動測試校正，不能只憑手腕 Rz 角度用猜的。
        """
        xc = (dx_px * dist_z) / self.fx
        yc = (dy_px * dist_z) / self.fy

        cos_t = np.cos(CAM_TO_BASE_YAW_RAD)
        sin_t = np.sin(CAM_TO_BASE_YAW_RAD)
        delta_x_base = xc * cos_t - yc * sin_t
        delta_y_base = xc * sin_t + yc * cos_t
        return float(delta_x_base), float(delta_y_base)

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray, estimate_tag_pose=False)
        center_x, center_y = int(self.cx), int(self.cy)

        cv2.line(cv_image, (center_x - 30, center_y), (center_x + 30, center_y), (255, 120, 0), 2)
        cv2.line(cv_image, (center_x, center_y - 30), (center_x, center_y + 30), (255, 120, 0), 2)

        target_tag = None
        target_info = None

        for tag in tags:
            corners = np.int32(tag.corners)
            cv2.polylines(cv_image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            t_center = (int(tag.center[0]), int(tag.center[1]))
            cv2.circle(cv_image, t_center, 5, (0, 0, 255), -1)

            dist_z = self.get_tag_analytic(tag)
            if dist_z is None:
                continue

            dx = t_center[0] - center_x
            dy = t_center[1] - center_y

            info_text = f"ID:{tag.tag_id} Z:{dist_z:.2f}m dX:{dx}px dY:{dy}px"
            cv2.putText(cv_image, info_text, (t_center[0] - 40, t_center[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            if tag.tag_id == 0:
                target_tag = tag
                target_info = (dx, dy, dist_z)

        now = time.time()
        status_text = "STAGE 1: SEARCHING FOR TAG..."
        status_color = (255, 255, 255)

        # ======================================================================
        # --- 階段 0：開機靜止預熱 ---
        # ======================================================================
        if self.stage == STAGE_0_WARMUP:
            if (now - self.start_node_time > 1.2) and (self.feedback_count >= 10) and (self.current_flange_pose_m is not None):
                self.stage = STAGE_1_SEARCH
                self.patrol_wait_start = now
                self.get_logger().info(">>> 【手臂狀態已就緒】開啟全局搜尋！ <<<")
            status_text = "STATUS: WARMING UP ROBOT POSE..."
            status_color = (0, 165, 255)

        # ======================================================================
        # --- 階段 4：鎖定完成 ---
        # ======================================================================
        elif self.stage == STAGE_4_LOCKED:
            if target_info is not None:
                dx, dy, _ = target_info
                err_px = np.sqrt(dx**2 + dy**2)
                status_text = f"STATUS: [LOCKED @ 30CM] RESIDUAL: {err_px:.1f}px"
            else:
                status_text = "STATUS: [LOCKED @ 30CM] BULLSEYE MATCHED!"
            status_color = (0, 255, 0)

        # ======================================================================
        # --- 階段 3：二次重捕獲＋閉環純直線強效微調 ---
        # ======================================================================
        elif self.stage == STAGE_3_FINE_SERVO:
            if self.is_stepping and (now - self.last_step_time > 0.85):
                self.is_stepping = False

            if target_info is not None:
                # 重新看到標籤：清空遺失計時／局部搜尋狀態，回到正常閉環微調
                self.lost_since_time = None
                self.reacquire_idx = 0
                self.reacquire_moving = False

                dx, dy, dist_z = target_info
                err_px = np.sqrt(dx**2 + dy**2)

                # 殘差 ≤ 18 像素 (約 1.5mm) 且連續 2 次停穩即鎖死
                if err_px <= 18.0:
                    self.bullseye_lock_count += 1
                    if self.bullseye_lock_count >= 2:
                        self.stage = STAGE_4_LOCKED
                        self.get_logger().info("==================================================")
                        self.get_logger().info(f">>> 【二次精修達成！藍十字壓死紅心】殘差: {err_px:.1f}px，永久鎖定！ <<<")
                        self.get_logger().info("==================================================")
                        status_text = f"STATUS: [LOCKED] BULLSEYE MATCHED! ({err_px:.1f}px)"
                        status_color = (0, 255, 0)
                    else:
                        status_text = f"STAGE 3: SETTLING ({self.bullseye_lock_count}/2)..."
                        status_color = (0, 255, 255)
                else:
                    self.bullseye_lock_count = 0
                    if not self.is_stepping and (now - self.last_step_time > 0.85):
                        self.step_toward_bullseye_smooth(dx, dy, dist_z, err_px)
                        self.last_step_time = now
                        self.is_stepping = True
                    status_text = f"STAGE 3: FINE CENTERING (Err: {err_px:.1f}px)..."
                    status_color = (0, 255, 255)
            else:
                self.bullseye_lock_count = 0

                if self.lost_since_time is None:
                    self.lost_since_time = now
                lost_elapsed = now - self.lost_since_time

                if lost_elapsed < REACQUIRE_GRACE_S:
                    status_text = "STAGE 3: RE-ACQUIRING TAG IN OVERHEAD VIEW..."
                    status_color = (0, 165, 255)
                elif self.reacquire_idx >= len(REACQUIRE_OFFSETS_M):
                    # 局部搜尋走完一輪都沒找到，放棄這個記憶點，回大範圍巡邏重新開始
                    self.get_logger().warn("局部搜尋仍找不到標籤，放棄記憶點，回到大範圍巡邏重新開始！")
                    self.stage = STAGE_1_SEARCH
                    self.memorized_target_hover_m = None
                    self.patrol_moving = False
                    self.patrol_wait_start = now
                    self.lost_since_time = None
                    self.reacquire_idx = 0
                    self.reacquire_moving = False
                    status_text = "STAGE 3: LOST TAG -> RESTART PATROL"
                    status_color = (0, 0, 255)
                else:
                    ox, oy = REACQUIRE_OFFSETS_M[self.reacquire_idx]
                    base_x, base_y, _ = self.memorized_target_hover_m
                    search_target = [base_x + ox, base_y + oy, REACQUIRE_SEARCH_Z_M]

                    if not self.reacquire_moving:
                        self.get_logger().info(
                            f"[局部搜尋] 拉高至 {REACQUIRE_SEARCH_Z_M}m 檢視偏移點 "
                            f"{self.reacquire_idx + 1}/{len(REACQUIRE_OFFSETS_M)}: ({ox:+.2f}, {oy:+.2f})"
                        )
                        self.send_motion(search_target, STRICT_90DEG_DOWNWARD_RPY, velocity_val=0.12, acc_time_val=0.35, motion_type=SetPositions.Request.LINE_T)
                        self.reacquire_moving = True
                        self.reacquire_move_time = now
                    elif now - self.reacquire_move_time > REACQUIRE_STEP_S:
                        self.reacquire_idx += 1
                        self.reacquire_moving = False

                    status_text = f"STAGE 3: LOCAL RE-SEARCH ({self.reacquire_idx + 1}/{len(REACQUIRE_OFFSETS_M)})..."
                    status_color = (0, 165, 255)

        # ======================================================================
        # --- 階段 2：直撲記憶點正上方 30cm ---
        # ======================================================================
        elif self.stage == STAGE_2_MEMORY_DASH:
            curr = np.array(self.current_flange_pose_m[:3])
            tgt = np.array(self.memorized_target_hover_m)
            rem_dist_m = np.linalg.norm(curr - tgt)
            elapsed = now - self.dash_start_time

            # 抵達記憶點 35mm 以內或飛行滿 2.5 秒，切入二次精修
            if rem_dist_m <= 0.035 or elapsed > 2.5:
                self.stage = STAGE_3_FINE_SERVO
                self.last_step_time = now
                self.is_stepping = False
                self.bullseye_lock_count = 0
                self.get_logger().info("==================================================")
                self.get_logger().info(">>> 【已抵達記憶點正上方】啟動二次重捕獲閉環微調！ <<<")
                self.get_logger().info("==================================================")
                status_text = "STAGE 2: ARRIVED -> FINE SERVO ACTIVE"
                status_color = (0, 255, 255)
            else:
                status_text = f"STAGE 2: MEMORY DASH TO 30CM (Remain: {rem_dist_m*1000.0:.1f}mm)..."
                status_color = (0, 255, 255)

        # ======================================================================
        # --- 階段 1：搜尋與初估記憶 ---
        # ======================================================================
        elif self.stage == STAGE_1_SEARCH:
            found_valid_target = False

            if target_tag is not None and self.current_flange_pose_m[0] > 0.05:
                dx, dy, dist_z = target_info
                px, py, _ = self.current_flange_pose_m[:3]

                delta_x_base, delta_y_base = self.cam_to_base_delta(dx, dy, dist_z)

                target_x = float(px + delta_x_base)
                target_y = float(py + delta_y_base)
                target_z = float(FINAL_HOVER_Z_M)

                # 工作區合理邊界檢查 (X: 0.08m ~ 0.70m, |Y| < 0.45m)
                if (target_x < 0.08 or target_x > 0.70) or abs(target_y) > 0.45:
                    self.get_logger().warn(f"計算目標點 [X:{target_x:.3f}, Y:{target_y:.3f}] 超界，忽略此次偵測，繼續巡邏！")
                else:
                    found_valid_target = True
                    self.memorized_target_hover_m = [target_x, target_y, target_z]
                    self.patrol_moving = False
                    self.stage = STAGE_2_MEMORY_DASH
                    self.dash_start_time = now

                    self.get_logger().info("==================================================")
                    self.get_logger().info(f">>> 【遠處初次捕獲！只算粗略位置，姿態不動】 <<<")
                    self.get_logger().info(f"粗目標: X={target_x:.4f}m, Y={target_y:.4f}m, Z={target_z:.4f}m")
                    self.get_logger().info("姿態全程鎖定 90 度鉛垂向下，僅做水平位置粗定位，不轉向！")
                    self.get_logger().info("==================================================")

                    # 用 LINE_T (卡氏直線) 取代 PTP_T：起訖姿態相同時 LINE_T 全程平滑
                    # 鎖死向下姿態，不會像 PTP_T 關節空間插值那樣中途看起來亂轉。
                    self.send_motion(self.memorized_target_hover_m, STRICT_90DEG_DOWNWARD_RPY, velocity_val=0.22, acc_time_val=0.45, motion_type=SetPositions.Request.LINE_T)
                    status_text = "STAGE 1: TAG MEMORIZED -> DASHING TO 30CM!"
                    status_color = (0, 255, 255)

            if not found_valid_target:
                # 尚未看到可用標籤 (沒看到 / 超界)：持續在航點間巡邏搜尋
                if (now - self.patrol_wait_start > 3.5) and self.patrol_moving:
                    self.patrol_moving = False
                    self.patrol_wait_start = now

                if self.patrol_moving:
                    if self.has_reached_patrol(self.SEARCH_WAYPOINTS_M_cur()):
                        self.patrol_moving = False
                        self.patrol_wait_start = now
                    status_text = f"STAGE 1: PATROL SCANNING WP {self.waypoint_idx + 1}"
                    status_color = (255, 200, 0)
                else:
                    if (now - self.patrol_wait_start) > 0.4:
                        target_pos = SEARCH_WAYPOINTS_M[self.waypoint_idx]
                        self.waypoint_idx = (self.waypoint_idx + 1) % len(SEARCH_WAYPOINTS_M)

                        self.patrol_moving = True
                        self.patrol_wait_start = now
                        self.get_logger().info(f"[巡邏] 移動至航點: {target_pos}")
                        # 同樣改用 LINE_T，巡邏移動全程保持姿態向下、不亂轉
                        self.send_motion(target_pos, STRICT_90DEG_DOWNWARD_RPY, velocity_val=0.18, acc_time_val=0.50, motion_type=SetPositions.Request.LINE_T)
                    status_text = "STAGE 1: SEARCHING FOR TAG..."
                    status_color = (255, 255, 255)

        cv2.putText(cv_image, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        if self.memorized_target_hover_m is not None:
            cv2.putText(cv_image, f"Target: [{self.memorized_target_hover_m[0]:.3f}, {self.memorized_target_hover_m[1]:.3f}]m",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.putText(cv_image, "Press 'r' to Reset | 'q' or 'ESC' to Exit",
                    (20, cv_image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        self.display_frame = cv_image

    def step_toward_bullseye_smooth(self, dx_px, dy_px, dist_z, err_px):
        """二次精修：根據殘差動態微調"""
        px, py, _ = self.current_flange_pose_m[:3]
        delta_x_base, delta_y_base = self.cam_to_base_delta(dx_px, dy_px, dist_z)

        # 動態分級步長與速度（提升低速下限，防止 TM 控制器拒絕）
        if err_px > 90.0:
            gain = 0.65
            max_step = 0.035
            vel = 0.08
        elif err_px > 35.0:
            gain = 0.45
            max_step = 0.015
            vel = 0.05
        else:
            gain = 0.25
            max_step = 0.005
            vel = 0.03

        step_x = float(np.clip(gain * delta_x_base, -max_step, max_step))
        step_y = float(np.clip(gain * delta_y_base, -max_step, max_step))

        next_pos = [
            float(px + step_x),
            float(py + step_y),
            FINAL_HOVER_Z_M
        ]

        self.get_logger().info(
            f"[二次閉環微修] 殘差:{err_px:.1f}px -> 平移 dX:{step_x*1000:+.1f}mm, dY:{step_y*1000:+.1f}mm (速度:{vel})"
        )
        self.send_motion(next_pos, STRICT_90DEG_DOWNWARD_RPY, velocity_val=vel, acc_time_val=0.25, motion_type=SetPositions.Request.LINE_T)

    def SEARCH_WAYPOINTS_M_cur(self):
        prev_idx = (self.waypoint_idx - 1) % len(SEARCH_WAYPOINTS_M)
        return SEARCH_WAYPOINTS_M[prev_idx]

    def has_reached_patrol(self, target_pos_m, threshold_m=0.040):
        if self.current_flange_pose_m is None or target_pos_m is None:
            return False
        curr = np.array(self.current_flange_pose_m[:3])
        target = np.array(target_pos_m[:3])
        return np.linalg.norm(curr - target) < threshold_m

    def send_motion(self, target_pos_m, target_rpy_rad, velocity_val=0.20, acc_time_val=0.35, motion_type=SetPositions.Request.PTP_T):
        req = SetPositions.Request()
        req.motion_type = motion_type
        req.positions = [
            float(target_pos_m[0]), float(target_pos_m[1]), float(target_pos_m[2]),
            float(target_rpy_rad[0]), float(target_rpy_rad[1]), float(target_rpy_rad[2])
        ]
        req.velocity = float(velocity_val)
        req.acc_time = float(acc_time_val)
        req.blend_percentage = 0
        req.fine_goal = True

        future = self.cli_set_pos.call_async(req)
        future.add_done_callback(self.service_response_callback)

    def service_response_callback(self, future):
        try:
            response = future.result()
            if not response.ok:
                self.get_logger().error(
                    "【TM 控制器拒絕執行】ok=False！請確認：\n"
                    "  1. TMflow 是否處於執行狀態 (綠色 Play)？\n"
                    "  2. 流程指標是否停在「Listen」節點？\n"
                    "  3. 若為 Manual 模式，是否有按住安全致動開關？"
                )
        except Exception as e:
            self.get_logger().error(f"服務異常: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = ArmMemoryGuidedAligner()

    cv2.namedWindow(node.win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(node.win_name, 848, 480)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.display_frame is not None:
                cv2.imshow(node.win_name, node.display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break
                elif key == ord('r'):
                    node.stage = STAGE_0_WARMUP
                    node.start_node_time = time.time()
                    node.patrol_moving = False
                    node.memorized_target_hover_m = None
                    node.is_stepping = False
                    node.bullseye_lock_count = 0
                    node.lost_since_time = None
                    node.reacquire_idx = 0
                    node.reacquire_moving = False
                    node.patrol_wait_start = time.time()
                    node.get_logger().info(">>> 已重置，重新開始搜尋... <<<")
            else:
                if (cv2.waitKey(10) & 0xFF) in [ord('q'), 27]:
                    break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
