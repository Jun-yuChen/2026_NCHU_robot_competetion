import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from dt_apriltags import Detector
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import CameraInfo, Image
from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SetPositions

# 跟 py_apriltag 系列腳本一致的 namespace 慣例（arm_find_tag.py 原本訂閱
# /camera/color/image_raw 是錯的，這裡改成正確的 /camera/camera/...）。
EIH_CAMERA_NS = '/camera/camera'

TAG_ID = 0
TAG_SIZE = 0.05                # 5 公分標籤，請實際拿尺確認跟 arm_find_tag.py 假設的一致
FINAL_HOVER_Z_M = 0.30         # 最終懸停高度 30cm
PATROL_SEARCH_Z_M = 0.40       # 高空巡邏高度 40cm

# 【90度垂直朝向桌面姿態基準】跟 arm_find_tag.py 一樣，STAGE_1 巡邏時沿用這個姿態
STRICT_90DEG_DOWNWARD_RPY = [3.1415926, 0.0, 2.4630]

# T_G_C 讀取路徑：故意用絕對路徑，不用相對路徑。
# 專案裡實際上有兩份同名的 ICA_Lab_UMI_Config.yaml，內容不一樣：
#   /ros2_ws/ICA_Lab_UMI_Config.yaml                              <- 這個 session 一直在用、已驗證的正版
#   /ros2_ws/src/py_gripper/calibration/ICA_Lab_UMI_Config_0804_1.yaml <- 8/4 的舊快照，T_G_C 數值不同
# 用相對路徑透過 `ros2 run` 啟動時 cwd 不保證是 /ros2_ws，容易誤讀到別份，所以寫死絕對路徑。
T_G_C_CONFIG_PATH = '/ros2_ws/ICA_Lab_UMI_Config.yaml'

# 巡邏航點 (m) — 跟 arm_find_tag.py 完全相同，維持搜尋階段行為不變
SEARCH_WAYPOINTS_M = [
    [0.32,  0.00, PATROL_SEARCH_Z_M],
    [0.32,  0.12, PATROL_SEARCH_Z_M],
    [0.32, -0.12, PATROL_SEARCH_Z_M],
    [0.42,  0.00, PATROL_SEARCH_Z_M],
]

# 安全邊界檢查，跟 arm_find_tag.py 一致
TARGET_X_RANGE = (0.08, 0.70)
TARGET_Y_ABS_MAX = 0.45

# 精修鎖定門檻：世界座標誤差 <= 3mm 且連續 2 次穩定才鎖死
FINE_LOCK_ERR_M = 0.003
FINE_LOCK_COUNT = 2
FINE_STEP_COOLDOWN_S = 0.85
# STAGE_3 連續多久偵測不到 tag（常見原因：夾爪本身擋住視野）才印警告，
# 純粹提醒用，不會自動切換狀態或觸發任何移動。
OCCLUSION_STUCK_WARN_S = 5.0

STAGE_0_WARMUP      = 0    # 開機靜止預熱 (防抽動)
STAGE_1_SEARCH       = 1    # 搜尋巡邏，任何姿態偵測到 tag 都能正確算出世界座標
STAGE_2_DASH          = 2    # 【記憶盲飛】：依實際量測到的 tag 世界座標直撲上方 30cm
STAGE_3_FINE_SERVO    = 3    # 到位後重新偵測，閉環微調消除殘差
STAGE_4_LOCKED        = 4    # 永久鎖死


class ArmDockApriltag(Node):
    """
    跟 arm_find_tag.py 目的相同（找到 tag_id=0，飛到正上方 30cm 懸停），
    但拿掉「視在尺寸推距離 + 寫死軸對應」那套只有在鏡頭精確垂直向下時才成立
    的土法煉鋼算法，改用已經校正驗證過的 T_G_C（讀 ICA_Lab_UMI_Config.yaml，
    不修改）搭配 dt_apriltags 的完整 PnP pose estimation，把 tag 在相機座標
    的 pose 轉成 base 座標系下的真實世界位置。這個轉換不管手臂當下是什麼
    姿態都是準的，也不會因為離 tag 遠近不同而放大誤差。
    """

    def __init__(self, T_G_C: np.ndarray):
        super().__init__('arm_dock_apriltag')
        self.T_G_C = T_G_C

        self.bridge = CvBridge()
        self.at_detector = Detector(
            searchpath=['apriltags'],
            families='tag36h11',
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        self.info = None
        self.cam_info_sub = self.create_subscription(
            CameraInfo, f'{EIH_CAMERA_NS}/color/camera_info',
            self.cam_info_callback, 10)
        self.sub_image = self.create_subscription(
            Image, f'{EIH_CAMERA_NS}/color/image_raw',
            self.image_callback, 1)

        # current_positions 直接視為 (m, rad)，不做 mm/度猜測轉換 —
        # 這點要跟 T_G_C 校正時的假設一致，T_G_C 就是用這個假設驗證過的。
        self.current_positions = None
        self.feedback_count = 0
        self.sub_feedback = self.create_subscription(
            FeedbackState, 'feedback_states', self.pos_callback, 10)

        self.cli_set_pos = self.create_client(SetPositions, 'set_positions')
        self.get_logger().info("等待手臂 /set_positions 服務上線...")
        while not self.cli_set_pos.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("尚未連上 TM Robot 服務...")

        self.start_node_time = time.time()
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
        self.last_tag_seen_time = None
        self.last_occlusion_warn_time = 0.0

        self.win_name = "RealSense Apriltag Dock (T_G_C accurate)"
        self.get_logger().info("【精確版記憶導航】啟動！")

    def cam_info_callback(self, msg: CameraInfo):
        self.info = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'ppx': msg.k[2],
            'ppy': msg.k[5],
        }
        self.destroy_subscription(self.cam_info_sub)

    def pos_callback(self, msg: FeedbackState):
        self.current_positions = list(msg.tool_pose)
        self.feedback_count += 1
        # 一次性檢查：T_G_C 是用「tool_pose 直接視為 (m, rad)，不做單位轉換」這個假設
        # 校正並驗證過的。這裡只印警告，不做任何轉換，如果看到警告要回頭確認假設是否成立。
        if self.feedback_count == 1:
            p = self.current_positions
            if any(abs(v) > 5.0 for v in p[:3]) or any(abs(v) > 7.0 for v in p[3:]):
                self.get_logger().warn(
                    f"tool_pose 數值看起來不像 (m, rad)：{p}，"
                    "T_G_C 是用直接當 m/rad 的假設校正驗證過的，這裡沒有做任何單位轉換，"
                    "如果實際是 mm/度，後面算出來的世界座標會整個錯掉，請先確認！"
                )

    def detectTag(self, frame):
        if self.info is None:
            return None, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        cam_params = [self.info['fx'], self.info['fy'], self.info['ppx'], self.info['ppy']]
        results = self.at_detector.detect(gray, True, cam_params, TAG_SIZE)
        for r in results:
            if r.tag_id != TAG_ID:
                continue
            T_C_A = np.eye(4)
            T_C_A[:3, :3] = r.pose_R
            T_C_A[:3, 3] = r.pose_t.reshape(3)
            return T_C_A, r
        return None, None

    def get_tag_world_pos(self, T_C_A: np.ndarray) -> np.ndarray:
        """T_W_G @ T_G_C @ T_C_A 的位移部分：tag 在 base 座標系的真實位置，
        不管手臂當下姿態是什麼都準確，取代舊版只在垂直向下時才成立的
        cam_to_base_delta 寫死軸對應。"""
        T_W_G = np.eye(4)
        T_W_G[:3, :3] = R.from_euler('xyz', self.current_positions[3:], degrees=False).as_matrix()
        T_W_G[:3, 3] = self.current_positions[:3]
        T_W_A = T_W_G @ self.T_G_C @ T_C_A
        return T_W_A[:3, 3]

    def has_reached(self, target_pos_m, threshold_m=0.040):
        if self.current_positions is None or target_pos_m is None:
            return False
        curr = np.array(self.current_positions[:3])
        target = np.array(target_pos_m[:3])
        return np.linalg.norm(curr - target) < threshold_m

    def step_toward_target(self, err_xy: np.ndarray, err_norm: float):
        """精修階段的閉環微調：直接在世界座標(m)上做比例控制加限幅，
        不用再像舊版那樣經過像素/視在尺寸換算。"""
        if err_norm > 0.05:
            gain, max_step, vel = 0.50, 0.020, 0.10
        elif err_norm > 0.02:
            gain, max_step, vel = 0.40, 0.010, 0.05
        else:
            gain, max_step, vel = 0.25, 0.003, 0.02

        step = np.clip(gain * err_xy, -max_step, max_step)
        px, py = self.current_positions[0], self.current_positions[1]
        next_pos = [float(px + step[0]), float(py + step[1]), FINAL_HOVER_Z_M]

        self.get_logger().info(
            f"[精修] 殘差:{err_norm*1000:.1f}mm -> 平移 dX:{step[0]*1000:+.1f}mm, dY:{step[1]*1000:+.1f}mm"
        )
        self.send_motion(next_pos, STRICT_90DEG_DOWNWARD_RPY, velocity_val=vel, acc_time_val=0.25,
                          motion_type=SetPositions.Request.LINE_T)

    def send_motion(self, target_pos_m, target_rpy_rad, velocity_val=0.20, acc_time_val=0.35,
                     motion_type=SetPositions.Request.PTP_T):
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
                self.get_logger().error(f"【TM 控制器拒絕執行】ok={response.ok}")
        except Exception as e:
            self.get_logger().error(f"服務異常: {str(e)}")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return
        now = time.time()
        status_text = "STAGE 1: SEARCHING FOR TAG..."
        status_color = (255, 255, 255)

        T_C_A, tag = self.detectTag(cv_image)
        if tag is not None:
            corners = np.int32(tag.corners)
            cv2.polylines(cv_image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            t_center = (int(tag.center[0]), int(tag.center[1]))
            cv2.circle(cv_image, t_center, 5, (0, 0, 255), -1)

        # ======================================================================
        if self.stage == STAGE_0_WARMUP:
            if (now - self.start_node_time > 1.2) and (self.feedback_count >= 10) and (self.current_positions is not None):
                self.stage = STAGE_1_SEARCH
                self.patrol_wait_start = now
                self.get_logger().info(">>> 【手臂初態已穩定】開啟全局搜尋！ <<<")
            status_text = "STATUS: WARMING UP ROBOT POSE..."
            status_color = (0, 165, 255)

        # ======================================================================
        elif self.stage == STAGE_4_LOCKED:
            status_text = "STATUS: [LOCKED @ 30CM] BULLSEYE MATCHED!"
            status_color = (0, 255, 0)

        # ======================================================================
        elif self.stage == STAGE_3_FINE_SERVO:
            if self.is_stepping and (now - self.last_step_time > FINE_STEP_COOLDOWN_S):
                self.is_stepping = False

            if T_C_A is not None:
                tag_world_pos = self.get_tag_world_pos(T_C_A)
                curr_xy = np.array(self.current_positions[:2])
                err_xy = tag_world_pos[:2] - curr_xy
                err_norm = float(np.linalg.norm(err_xy))

                self.last_tag_seen_time = now
                if err_norm <= FINE_LOCK_ERR_M:
                    self.bullseye_lock_count += 1
                    if self.bullseye_lock_count >= FINE_LOCK_COUNT:
                        self.stage = STAGE_4_LOCKED
                        self.get_logger().info(f">>> 【精修完成】殘差: {err_norm*1000:.1f}mm，永久鎖死！ <<<")
                        status_text = f"STATUS: [LOCKED] BULLSEYE MATCHED! ({err_norm*1000:.1f}mm)"
                        status_color = (0, 255, 0)
                    else:
                        status_text = f"STAGE 3: SETTLING ({self.bullseye_lock_count}/{FINE_LOCK_COUNT})..."
                        status_color = (0, 255, 255)
                else:
                    self.bullseye_lock_count = 0
                    if not self.is_stepping and (now - self.last_step_time > FINE_STEP_COOLDOWN_S):
                        self.step_toward_target(err_xy, err_norm)
                        self.last_step_time = now
                        self.is_stepping = True
                    status_text = f"STAGE 3: FINE CENTERING (Err: {err_norm*1000:.1f}mm)..."
                    status_color = (0, 255, 255)
            else:
                # 單幀沒偵測到（常見原因：夾爪本身擋到 tag）不代表沒對準，
                # 手臂這裡也沒有送任何移動指令、維持原地，所以不重置 bullseye_lock_count，
                # 避免夾爪偶爾入鏡就讓精修進度整個歸零、永遠鎖不了。
                if self.last_tag_seen_time is not None and (now - self.last_tag_seen_time > OCCLUSION_STUCK_WARN_S):
                    if now - self.last_occlusion_warn_time > OCCLUSION_STUCK_WARN_S:
                        self.get_logger().warn(
                            f"連續 {now - self.last_tag_seen_time:.1f}s 偵測不到 tag，"
                            "如果不是暫時被夾爪擋到，可能是這個角度/高度看不到，考慮調整 HOVER_HEIGHT_M 或懸停位置"
                        )
                        self.last_occlusion_warn_time = now
                status_text = "STAGE 3: RE-ACQUIRING TAG IN OVERHEAD VIEW..."
                status_color = (0, 165, 255)

        # ======================================================================
        elif self.stage == STAGE_2_DASH:
            curr = np.array(self.current_positions[:3])
            tgt = np.array(self.memorized_target_hover_m)
            rem_dist_m = np.linalg.norm(curr - tgt)
            elapsed = now - self.dash_start_time

            if rem_dist_m <= 0.040 or elapsed > 2.2:
                self.stage = STAGE_3_FINE_SERVO
                self.last_step_time = now
                self.is_stepping = False
                self.bullseye_lock_count = 0
                self.last_tag_seen_time = now
                self.last_occlusion_warn_time = now
                self.get_logger().info(">>> 【已抵達記憶點正上方 30cm】開啟精修！ <<<")
                status_text = "STAGE 2: ARRIVED -> RE-ACQUIRING & FINE ALIGN"
                status_color = (0, 255, 255)
            else:
                status_text = f"STAGE 2: MEMORY DASH TO 30CM (Remain: {rem_dist_m*1000.0:.1f}mm)..."
                status_color = (0, 255, 255)

        # ======================================================================
        elif self.stage == STAGE_1_SEARCH:
            # 不管手臂當下是什麼姿態，只要偵測到 TAG_ID 就用 T_G_C 換算出正確的世界座標。
            if T_C_A is not None and self.current_positions is not None:
                tag_world_pos = self.get_tag_world_pos(T_C_A)
                target_x, target_y = float(tag_world_pos[0]), float(tag_world_pos[1])
                target_z = float(FINAL_HOVER_Z_M)

                if (target_x < TARGET_X_RANGE[0] or target_x > TARGET_X_RANGE[1]) or abs(target_y) > TARGET_Y_ABS_MAX:
                    self.get_logger().warn(f"計算目標點 [X:{target_x:.3f}, Y:{target_y:.3f}] 超界，略過！")
                else:
                    self.memorized_target_hover_m = [target_x, target_y, target_z]
                    self.patrol_moving = False
                    self.stage = STAGE_2_DASH
                    self.dash_start_time = now

                    self.get_logger().info("==================================================")
                    self.get_logger().info(f">>> 【捕獲標籤，T_G_C 換算世界座標】X={target_x:.4f}m, Y={target_y:.4f}m, Z={target_z:.4f}m")
                    self.get_logger().info("==================================================")

                    self.send_motion(self.memorized_target_hover_m, STRICT_90DEG_DOWNWARD_RPY,
                                      velocity_val=0.20, acc_time_val=0.50,
                                      motion_type=SetPositions.Request.PTP_T)
                    status_text = "STAGE 1: TAG MEMORIZED -> DASHING TO 30CM!"
                    status_color = (0, 255, 255)
            else:
                if (now - self.patrol_wait_start > 3.5) and self.patrol_moving:
                    self.patrol_moving = False
                    self.patrol_wait_start = now

                if self.patrol_moving:
                    prev_idx = (self.waypoint_idx - 1) % len(SEARCH_WAYPOINTS_M)
                    if self.has_reached(SEARCH_WAYPOINTS_M[prev_idx]):
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
                        self.send_motion(target_pos, STRICT_90DEG_DOWNWARD_RPY,
                                          velocity_val=0.18, acc_time_val=0.50,
                                          motion_type=SetPositions.Request.PTP_T)
                    status_text = "STAGE 1: SEARCHING FOR TAG..."
                    status_color = (255, 255, 255)

        cv2.putText(cv_image, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        if self.memorized_target_hover_m is not None:
            cv2.putText(cv_image, f"Memory Target: [{self.memorized_target_hover_m[0]:.3f}, {self.memorized_target_hover_m[1]:.3f}]m",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(cv_image, "Press 'r' to Reset | 'q' or 'ESC' to Exit",
                    (20, cv_image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        self.display_frame = cv_image


def main(args=None):
    print(f"讀取 T_G_C 來源: {T_G_C_CONFIG_PATH}")
    with open(T_G_C_CONFIG_PATH, 'r') as f:
        config_data = yaml.safe_load(f)
    T_G_C = np.array(config_data['T_G_C'])
    print("T_G_C:")
    print(np.array2string(T_G_C, separator=',', precision=6))

    rclpy.init(args=args)
    node = ArmDockApriltag(T_G_C)

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
                    node.last_tag_seen_time = None
                    node.last_occlusion_warn_time = 0.0
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
