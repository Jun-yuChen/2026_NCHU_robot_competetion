import time
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import Image
from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SetPositions, SetIO

# ============================================================
# 目前這支程式只做一件事：測試「大 AprilTag 粗定位 -> 小 AprilTag 精定位」
# 這個視覺鎖定流程，還沒接上『從主機拔線、放回工作台孔位』的完整動作。
#
# 背景：工作台旁立了一根校正架，架上貼了兩顆 tag（實機照片確認過）：
#   - 大 tag：ID 0，邊長 13cm，從較遠處就能看到，用來粗定位。
#   - 小 tag：ID 0（跟大 tag 同一個 ID，用畫面上的像素大小分辨：同一幀
#     裡兩顆都看到時，像素比較大的是大 tag、比較小的是小 tag），邊長
#     4.75cm，只有靠近之後才看得到，用來精定位。
#
# 流程：操作者先用 TMflow 把手臂大致擺到校正架前方（這步驟目前不是這支
# 程式自動做的），接著輸入 find 指令，程式會：
#   1. 以目前位置為中心，小範圍移動搜尋大 tag。
#   2. 找到後，閉環微調 X/Y 把大 tag 置中。
#   3. 沿工具座標 Z 軸小步（1.5cm/步）往前靠近，每一步都先確認 tag 真的
#      變大了才繼續走，沒有變大或整個跟丟就立刻停下——絕不盲目往前衝。
#      直到畫面同時看到大小兩顆 tag，代表小 tag 已進入視野。
#   4. 換成用小 tag 做更精細的閉環置中。
#   5. 回報最終鎖定的位置，供之後接上「小 tag -> 各孔位固定偏移量」使用
#      （這個偏移量目前還沒量測，是下一步）。
#
# 工具座標 Z 軸到底是『靠近』還是『遠離』目標，目前沒有實測驗證過，所以
# 用經驗性的方式處理：每走一步就檢查 tag 有沒有變大，沒變大就直接中止，
# 不會自動反向硬闖。第一次實機測試請在旁邊盯著，方向不對就 Ctrl+C，
# 然後把 FORWARD_AXIS_SIGN 改成 -1.0 再試一次。
# ============================================================

# 【工作台固定座標】跟 arm_goto_fixed_table.py 同一組量測值，先保留備用，
# 目前這支程式的 find 指令還沒用到，等大小 tag 定位驗證過、量出 tag 到各
# 孔位的固定偏移量之後，才會接上「放線回工作台」的實際插入動作。
TABLE_POSES_M = {
    1: [0.47699, 0.23261, 0.42229, 1.569575, -0.021118, 1.668360],
    2: [0.47661, 0.23992, 0.32088, 1.565037, -0.017802, 1.608495],
    3: [0.47683, 0.21871, 0.23749, 1.626123, 0.033685, 1.689479],
}
APPROACH_POSES_M = {
    1: [0.39699, 0.23261, 0.42229, 1.569575, -0.021118, 1.668360],
    2: [0.33499, 0.20246, 0.34226, 1.663299, 0.067719, 1.725607],
    3: [0.39004, 0.21871, 0.23749, 1.626123, 0.033685, 1.689479],
}
HOST_FRONT_POSE_M = [0.37182, -0.15544, 0.42301, 1.638515, 0.015533, 1.536239]

ARRIVE_THRESHOLD_M = 0.005
RETREAT_DISTANCE_M = 0.10

GRIPPER_OPEN_STATE  = 0.0
GRIPPER_CLOSE_STATE = 1.0
GRIPPER_ACTUATION_DELAY_S = 1.0

# ------------------------------------------------------------
# 【RealSense + AprilTag 參數】相機模型沿用 arm_find_tag.py，實機請務必
# 重新校正，不要直接沿用這裡的預設值。
# ------------------------------------------------------------
CAMERA_K = np.array([
    [615.0,   0.0, 424.0],
    [  0.0, 615.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

REF_TAG_ID = 0            # 大、小 tag 用的是同一個 ID，靠像素大小分辨
BIG_TAG_SIZE_M = 0.13      # 大 tag 實際邊長
SMALL_TAG_SIZE_M = 0.0475  # 小 tag 實際邊長

# 【相機像素軸 -> 底座 X/Y 軸的旋轉角】跟 arm_find_tag.py 同一顆相機、
# 同一個安裝方式才能直接沿用；如果相機角度不同，務必重新點動校正。
CAM_TO_BASE_YAW_RAD = -np.pi / 2.0

# 找大 tag 用的小範圍搜尋偏移量（以 find 指令當下的位置為中心），由近到遠。
LOCAL_SEARCH_OFFSETS_M = [
    (0.00, 0.00),
    (0.04, 0.00), (-0.04, 0.00), (0.00, 0.04), (0.00, -0.04),
    (0.04, 0.04), (-0.04, -0.04), (0.04, -0.04), (-0.04, 0.04),
    (0.08, 0.00), (-0.08, 0.00), (0.00, 0.08), (0.00, -0.08),
]
LOCAL_SEARCH_STEP_SETTLE_S = 0.6

# 沿工具座標 Z 軸每步靠近的距離、最多靠近的總距離（超過還沒看到小 tag 就放棄）。
FORWARD_AXIS_SIGN = 1.0
FORWARD_STEP_M = 0.015
FORWARD_MAX_TRAVEL_M = 0.20
FORWARD_STEP_SETTLE_S = 0.4

ALIGN_TOLERANCE_PX  = 15.0   # 判定「已置中」的像素殘差門檻
ALIGN_LOCK_COUNT     = 3     # 連續幾次都在門檻內，才真的判定對準（濾掉雜訊）
ALIGN_TIMEOUT_S       = 8.0
ALIGN_TAG_LOST_TIMEOUT_S = 2.0

ALIGN_STEP_GAIN    = 0.4
ALIGN_STEP_MAX_M    = 0.006
ALIGN_STEP_VELOCITY = 0.03
ALIGN_STEP_ACC_TIME  = 0.25
ALIGN_STEP_SETTLE_S   = 0.5


class ArmReturnToTable(Node):
    def __init__(self):
        super().__init__('arm_return_to_table')
        self.bridge = CvBridge()
        self.detector = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0)
        self.fx, self.fy = CAMERA_K[0, 0], CAMERA_K[1, 1]
        self.cx, self.cy = CAMERA_K[0, 2], CAMERA_K[1, 2]

        self.current_flange_pose_m = None   # [x, y, z, rx, ry, rz]，rpy 單位 rad
        self.latest_tag_candidates = []      # 這一幀看到的所有 REF_TAG_ID: [{dx,dy,s_px,t}, ...]

        self.win_name = "RealSense Tag Search (arm_return_to_table)"
        self.display_frame = None

        self.sub_feedback = self.create_subscription(FeedbackState, 'feedback_states', self.feedback_callback, 10)
        self.sub_image = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 1)
        self.cli_set_pos = self.create_client(SetPositions, 'set_positions')
        self.cli_set_io = self.create_client(SetIO, '/set_io')

        self.get_logger().info("等待手臂 /set_positions 服務上線...")
        while not self.cli_set_pos.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("尚未連上 TM Robot 服務...")

        self.get_logger().info("等待夾爪 /set_io 服務上線...")
        while not self.cli_set_io.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("尚未連上夾爪 IO 服務...")

        self.get_logger().info("服務已就緒！")

    def feedback_callback(self, msg: FeedbackState):
        raw_pose = msg.tool_pose
        if len(raw_pose) < 6:
            return
        p0, p1, p2 = raw_pose[0], raw_pose[1], raw_pose[2]
        if abs(p0) > 5.0 or abs(p1) > 5.0 or abs(p2) > 5.0:
            pos_m = [float(p0 / 1000.0), float(p1 / 1000.0), float(p2 / 1000.0)]
        else:
            pos_m = [float(p0), float(p1), float(p2)]

        # TM 驅動有時回傳角度、有時回傳弧度，跟 arm_find_tag.py 一樣用數值大小粗略判斷
        rpy = [
            float(np.deg2rad(raw_pose[3])) if abs(raw_pose[3]) > 7.0 else float(raw_pose[3]),
            float(np.deg2rad(raw_pose[4])) if abs(raw_pose[4]) > 7.0 else float(raw_pose[4]),
            float(np.deg2rad(raw_pose[5])) if abs(raw_pose[5]) > 7.0 else float(raw_pose[5]),
        ]
        self.current_flange_pose_m = pos_m + rpy

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray, estimate_tag_pose=False)
        cx_i, cy_i = int(self.cx), int(self.cy)

        cv2.line(cv_image, (cx_i - 30, cy_i), (cx_i + 30, cy_i), (255, 120, 0), 2)
        cv2.line(cv_image, (cx_i, cy_i - 30), (cx_i, cy_i + 30), (255, 120, 0), 2)

        candidates = []
        for tag in tags:
            is_ref = (tag.tag_id == REF_TAG_ID)
            corners = np.int32(tag.corners)
            cv2.polylines(cv_image, [corners], isClosed=True,
                          color=(0, 255, 0) if is_ref else (0, 140, 255), thickness=2)
            t_center = (int(tag.center[0]), int(tag.center[1]))
            cv2.circle(cv_image, t_center, 5, (0, 0, 255), -1)
            cv2.putText(cv_image, f"ID:{tag.tag_id}", (t_center[0] - 20, t_center[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if is_ref:
                s_px = self.tag_pixel_size(tag)
                if s_px is None:
                    continue
                dx = float(t_center[0] - self.cx)
                dy = float(t_center[1] - self.cy)
                candidates.append({'dx': dx, 'dy': dy, 's_px': s_px, 't': time.time()})
                cv2.putText(cv_image, f"{s_px:.0f}px", (t_center[0] - 20, t_center[1] + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.latest_tag_candidates = candidates

        if candidates:
            sizes = ", ".join(f"{c['s_px']:.0f}px" for c in sorted(candidates, key=lambda c: -c['s_px']))
            status = f"SEE tag ID {REF_TAG_ID} x{len(candidates)} ({sizes})"
            status_color = (0, 255, 0)
        else:
            status = f"NO tag ID {REF_TAG_ID} visible"
            status_color = (0, 165, 255)
        cv2.putText(cv_image, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        self.display_frame = cv_image

    def tag_pixel_size(self, tag):
        c = tag.corners
        s_px = (np.linalg.norm(c[1] - c[0]) +
                np.linalg.norm(c[2] - c[1]) +
                np.linalg.norm(c[3] - c[2]) +
                np.linalg.norm(c[0] - c[3])) / 4.0
        if s_px < 10.0:
            return None
        return s_px

    def dist_z_for(self, s_px, real_size_m):
        return (self.fx * real_size_m) / s_px

    def pick_candidate(self, prefer='largest'):
        cands = self.latest_tag_candidates
        if not cands:
            return None
        if prefer == 'largest':
            return max(cands, key=lambda c: c['s_px'])
        return min(cands, key=lambda c: c['s_px'])

    def cam_to_base_delta(self, dx_px, dy_px, dist_z):
        """像素平面偏移 -> 底座 X/Y 偏移 (m)，換算方式跟 arm_find_tag.py 相同。"""
        xc = (dx_px * dist_z) / self.fx
        yc = (dy_px * dist_z) / self.fy

        cos_t = np.cos(CAM_TO_BASE_YAW_RAD)
        sin_t = np.sin(CAM_TO_BASE_YAW_RAD)
        delta_x_base = xc * cos_t - yc * sin_t
        delta_y_base = xc * sin_t + yc * cos_t
        return float(delta_x_base), float(delta_y_base)

    def forward_vector_base_frame(self, sign=FORWARD_AXIS_SIGN):
        """目前工具姿態的局部 Z 軸，換算到底座座標系的方向向量（單位向量）。"""
        if self.current_flange_pose_m is None or len(self.current_flange_pose_m) < 6:
            return None
        rpy = self.current_flange_pose_m[3:6]
        R_cur = Rotation.from_euler('xyz', rpy).as_matrix()
        return sign * R_cur[:, 2]

    def send_motion(self, target_pos_m, target_rpy_rad, velocity_val=0.15, acc_time_val=0.4,
                     motion_type=SetPositions.Request.LINE_T):
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
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            self.get_logger().error("送出移動指令逾時，未收到服務回應！")
            return False

        response = future.result()
        if response is not None and not response.ok:
            self.get_logger().error(
                "【TM 控制器拒絕執行】ok=False！請確認 TMflow 是否處於 Play 狀態、"
                "流程指標停在 Listen 節點、或手動模式的安全致動開關。"
            )
            return False
        return True

    def set_gripper(self, state):
        """state = 0.0 打開夾爪放線, state = 1.0 閉合夾爪夾住線"""
        req = SetIO.Request()
        req.module = 1
        req.type = 1
        req.pin = 0
        req.state = float(state)

        future = self.cli_set_io.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if not future.done() or future.result() is None:
            self.get_logger().error("夾爪 IO 指令逾時，未收到服務回應！")
            return False
        return True

    def refresh_display(self):
        if self.display_frame is not None:
            cv2.imshow(self.win_name, self.display_frame)
            cv2.waitKey(1)

    def settle_with_display(self, duration_s):
        end_time = time.time() + duration_s
        while rclpy.ok() and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)
            self.refresh_display()

    def wait_until_arrived(self, target_pos_m, threshold_m=ARRIVE_THRESHOLD_M):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            self.refresh_display()
            if self.current_flange_pose_m is not None:
                dist = sum((self.current_flange_pose_m[i] - target_pos_m[i]) ** 2 for i in range(3)) ** 0.5
                if dist <= threshold_m:
                    return True
        return False

    def local_search_for_tag(self, base_pos_m, base_rpy_rad, offsets=LOCAL_SEARCH_OFFSETS_M,
                             per_step_wait_s=LOCAL_SEARCH_STEP_SETTLE_S):
        """以 base_pos_m 為中心，依 offsets 小範圍移動找大 tag，找到就停在該點。

        只負責『把 tag 帶進畫面』，不做精細對準（精細對準交給 center_on_tag）。
        回傳 (found: bool, reason: str)。
        """
        for ox, oy in offsets:
            if not rclpy.ok():
                return False, "節點已關閉"
            probe_pos = [base_pos_m[0] + ox, base_pos_m[1] + oy, base_pos_m[2]]
            self.get_logger().info(f"[搜尋] 移動到偏移點 ({ox * 100:+.0f}, {oy * 100:+.0f})cm 尋找大 tag...")
            if not self.send_motion(probe_pos, base_rpy_rad, velocity_val=0.08, acc_time_val=0.3):
                return False, "搜尋移動指令被拒絕"
            self.wait_until_arrived(probe_pos, threshold_m=0.005)
            self.settle_with_display(per_step_wait_s)
            if self.pick_candidate('largest') is not None:
                self.get_logger().info(f"[搜尋] 在偏移點 ({ox * 100:+.0f}, {oy * 100:+.0f})cm 找到大 tag！")
                return True, "找到"
        return False, "小範圍搜尋完仍未找到大 tag"

    def center_on_tag(self, hover_pos_m, hover_rpy_rad, real_size_m, prefer='largest',
                      tolerance_px=ALIGN_TOLERANCE_PX, lock_count_needed=ALIGN_LOCK_COUNT,
                      timeout_s=ALIGN_TIMEOUT_S, lost_timeout_s=ALIGN_TAG_LOST_TIMEOUT_S,
                      max_shift_m=0.12):
        """在 hover_pos_m 附近閉環微調 X/Y，把 pick_candidate(prefer) 選到的 tag 置中。

        回傳 (success, final_xy | None, reason)。
        """
        start_time = time.time()
        last_seen_time = start_time
        lock_count = 0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            self.refresh_display()
            now = time.time()

            if now - start_time > timeout_s:
                return False, None, f"置中逾時 ({timeout_s:.1f}s)"
            if self.current_flange_pose_m is None:
                continue

            cand = self.pick_candidate(prefer)
            if cand is None:
                if now - last_seen_time > lost_timeout_s:
                    return False, None, f"持續 {lost_timeout_s:.1f}s 看不到 tag"
                continue
            last_seen_time = now

            dx, dy, s_px = cand['dx'], cand['dy'], cand['s_px']
            dist_z = self.dist_z_for(s_px, real_size_m)
            err_px = math.hypot(dx, dy)

            cur_xy = np.array(self.current_flange_pose_m[:2])
            hover_xy = np.array(hover_pos_m[:2])
            total_shift = float(np.linalg.norm(cur_xy - hover_xy))
            if total_shift > max_shift_m:
                return False, None, (
                    f"置中位移量已達 {total_shift * 1000:.1f}mm，超過上限 "
                    f"{max_shift_m * 1000:.1f}mm，疑似認錯目標，放棄"
                )

            if err_px <= tolerance_px:
                lock_count += 1
                self.get_logger().info(f"[置中] 殘差 {err_px:.1f}px 已在門檻內 ({lock_count}/{lock_count_needed})")
                if lock_count >= lock_count_needed:
                    return True, self.current_flange_pose_m[:2], "已置中"
                self.settle_with_display(ALIGN_STEP_SETTLE_S)
                continue

            lock_count = 0
            delta_x_base, delta_y_base = self.cam_to_base_delta(dx, dy, dist_z)
            step_x = float(np.clip(ALIGN_STEP_GAIN * delta_x_base, -ALIGN_STEP_MAX_M, ALIGN_STEP_MAX_M))
            step_y = float(np.clip(ALIGN_STEP_GAIN * delta_y_base, -ALIGN_STEP_MAX_M, ALIGN_STEP_MAX_M))
            next_pos = [
                self.current_flange_pose_m[0] + step_x,
                self.current_flange_pose_m[1] + step_y,
                hover_pos_m[2],
            ]
            self.get_logger().info(
                f"[置中] 殘差 {err_px:.1f}px -> 微調 dX:{step_x * 1000:+.1f}mm, dY:{step_y * 1000:+.1f}mm"
            )
            if not self.send_motion(next_pos, hover_rpy_rad, velocity_val=ALIGN_STEP_VELOCITY,
                                     acc_time_val=ALIGN_STEP_ACC_TIME):
                return False, None, "微調指令被控制器拒絕"
            self.wait_until_arrived(next_pos, threshold_m=0.003)
            self.settle_with_display(ALIGN_STEP_SETTLE_S)
        return False, None, "節點已關閉"

    def approach_forward_until_dual_tag(self, hover_rpy_rad, step_m=FORWARD_STEP_M,
                                        max_travel_m=FORWARD_MAX_TRAVEL_M):
        """沿工具局部 Z 軸小步靠近，直到畫面同時看到大小兩顆 tag。

        每一步都先確認 tag 真的變大了才繼續走；沒有變大、跟丟、或累積距離超過
        max_travel_m 都會立刻停下，不會盲目往前衝。回傳 (success, reason)。
        """
        cand = self.pick_candidate('largest')
        if cand is None:
            return False, "起始沒有偵測到 tag，無法開始靠近"
        last_s_px = cand['s_px']
        traveled = 0.0

        while traveled < max_travel_m:
            if not rclpy.ok():
                return False, "節點已關閉"
            if self.current_flange_pose_m is None:
                return False, "沒有手臂回饋位置"

            fwd = self.forward_vector_base_frame()
            if fwd is None:
                return False, "沒有姿態資訊，無法計算前進方向"

            next_pos = [
                self.current_flange_pose_m[0] + fwd[0] * step_m,
                self.current_flange_pose_m[1] + fwd[1] * step_m,
                self.current_flange_pose_m[2] + fwd[2] * step_m,
            ]
            self.get_logger().info(f"[靠近] 沿工具 Z 軸前進 {step_m * 100:.1f}cm -> {[round(v, 4) for v in next_pos]}")
            if not self.send_motion(next_pos, hover_rpy_rad, velocity_val=0.05, acc_time_val=0.25):
                return False, "靠近移動指令被拒絕"
            self.wait_until_arrived(next_pos, threshold_m=0.004)
            self.settle_with_display(FORWARD_STEP_SETTLE_S)
            traveled += step_m

            if len(self.latest_tag_candidates) >= 2:
                self.get_logger().info("[靠近] 同時看到兩顆 tag，小 tag 已進入視野！")
                return True, "同時偵測到大小兩顆 tag"

            cand = self.pick_candidate('largest')
            if cand is None:
                return False, "靠近途中跟丟了 tag，中止（可能方向錯誤或已經太近）"
            if cand['s_px'] <= last_s_px:
                return False, (
                    f"往前移動後 tag 沒有變大 ({last_s_px:.0f}px -> {cand['s_px']:.0f}px)，"
                    "方向可能反了，中止，請確認 FORWARD_AXIS_SIGN"
                )
            last_s_px = cand['s_px']

        return False, f"已靠近 {max_travel_m * 100:.0f}cm 仍未看到第二顆 tag，中止"

    def locate_reference_stand(self, base_pos_m, base_rpy_rad):
        """完整流程：搜尋大 tag -> 置中大 tag -> 往前靠近找小 tag -> 精細置中小 tag。

        回傳 (success, final_xy | None, reason)。
        """
        found, reason = self.local_search_for_tag(base_pos_m, base_rpy_rad)
        if not found:
            return False, None, f"搜尋大 tag 失敗：{reason}"

        if self.current_flange_pose_m is None:
            return False, None, "沒有手臂回饋位置"
        search_landing_pos = self.current_flange_pose_m[:3]

        self.get_logger().info("開始置中大 tag...")
        ok, xy, reason = self.center_on_tag(search_landing_pos, base_rpy_rad, BIG_TAG_SIZE_M,
                                            prefer='largest', max_shift_m=0.12)
        if not ok:
            return False, None, f"置中大 tag 失敗：{reason}"

        self.get_logger().info("大 tag 已置中，開始往前靠近尋找小 tag...")
        ok, reason = self.approach_forward_until_dual_tag(base_rpy_rad)
        if not ok:
            return False, None, f"往前靠近失敗：{reason}"

        if self.current_flange_pose_m is None:
            return False, None, "沒有手臂回饋位置"
        approach_landing_pos = self.current_flange_pose_m[:3]

        self.get_logger().info("開始精細置中小 tag...")
        ok, xy, reason = self.center_on_tag(approach_landing_pos, base_rpy_rad, SMALL_TAG_SIZE_M,
                                            prefer='smallest', tolerance_px=10.0, max_shift_m=0.05)
        if not ok:
            return False, None, f"精細置中小 tag 失敗：{reason}"

        return True, xy, "大小 tag 都已鎖定"


def main(args=None):
    rclpy.init(args=args)
    node = ArmReturnToTable()

    cv2.namedWindow(node.win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(node.win_name, 848, 480)

    while rclpy.ok() and node.current_flange_pose_m is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        node.refresh_display()

    node.get_logger().info(
        "已就緒。請先用 TMflow 把手臂大致擺到校正架前方，再輸入 find 開始搜尋、"
        "置中大 tag、往前靠近、精細置中小 tag。輸入 q 離開。"
    )
    node.get_logger().warn(
        "放線回工作台 1/2/3 孔位的完整動作還沒接上這套新的視覺定位（要先量出 tag "
        "到各孔位的固定偏移量），目前只支援 find 這個測試指令。"
    )

    try:
        while rclpy.ok():
            try:
                choice = input("指令 (find / q 離開): ").strip().lower()
            except EOFError:
                break

            if choice == 'q':
                break

            if choice != 'find':
                print("目前只支援 find 指令，或輸入 q 離開。")
                continue

            if node.current_flange_pose_m is None or len(node.current_flange_pose_m) < 6:
                print("還沒收到完整的手臂位置/姿態回饋，稍後再試。")
                continue

            base_pos = node.current_flange_pose_m[:3]
            base_rpy = node.current_flange_pose_m[3:6]
            node.get_logger().info(
                f"開始搜尋，基準位置: {[round(v, 4) for v in base_pos]}，"
                f"姿態: {[round(v, 4) for v in base_rpy]}（維持不變，全程只調整 X/Y）"
            )

            ok, xy, reason = node.locate_reference_stand(base_pos, base_rpy)
            if ok:
                node.get_logger().info(f">>> 鎖定成功！最終 X/Y: {[round(v, 4) for v in xy]} <<<")
            else:
                node.get_logger().error(f">>> 鎖定失敗：{reason} <<<")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
