"""Stage3導納插入的剛性搜尋版本——跟insertion_node_zero.py是平行檔案，
不修改舊版，兩支保留互相比較。

跟insertion_node_zero.py最大的差異，在Approaching/Searching這兩個階段：

    1. Approaching：完全剛性(rigid)直線下降，不套用導納——xc永遠等於xd，
       碰到力不會有任何柔順讓步。理由：下降過程中不希望任何力量(含雜訊)
       造成位置偏移，只在達到contact_force_threshold那一刻才轉換行為。
    2. Searching：X/Y走剛性螺旋軌跡(同樣不套用導納，直接等於spiral_
       trajectory算出來的位置)，Z軸改用force_hold_z_velocity()這個簡單
       比例力控制(不是導納)去維持search_desired_force的貼壓力——刻意
       不用insertion_node_zero.py那套admittance_step+F_desired的機制，
       是因為那套機制曾經實測出「收斂動態(~1秒)跟螺旋週期(~3.14秒)搭
       不上、互相干擾造成持續小幅震盪」的問題(見2026-08-20的調參紀錄)，
       這裡改用沒有二階動態、不會跟旋轉週期共振的簡單比例控制迴避掉
       這個問題。

Aligning(Step3)/Inserting(Step4)這兩個階段完全沿用insertion_node_zero.py
的邏輯(admittance_step+insertion_step34.py)不變——這兩個階段需要真正的
導納柔順，讓接頭被孔口導角被動導正，跟前兩階段的設計目標不同，不適用
「不用導納」的原則。

歸零機制、R_G_S處理方式都跟insertion_node_zero.py相同，理由見該檔案
開頭說明，這裡不重複贅述。
"""
import csv
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger

from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SendScript, SetPositions

from stage3_admittance.admittance_core import (
    admittance_step,
    blend_velocity,
    euler_to_matrix,
    filter_wrench,
    force_hold_z_velocity,
    linear_insertion_trajectory,
    spiral_trajectory,
)
from stage3_admittance.safety import (
    apply_deadband,
    check_force_limit,
    detect_force_rate_drop,
    detect_force_rate_rise,
    saturate_vector,
    workspace_repulsion,
)
from stage3_admittance.insertion_step34 import (
    aligning_trajectory,
    check_step3_false_positive,
)

ZERO_SETTLE_SEC = 0.3


class InsertionNodeRigidSearch(Node):
    def __init__(self):
        super().__init__('insertion_node_rigid_search')
        self._declare_params()

        self._latest_wrench_raw = None
        self._latest_tool_pose = None
        self._latest_feedback_msg = None

        self._active = False
        self._t0 = None
        self._start_pose_base = None
        self._R_start_to_base = None
        self._start_pose_insertion_frame = None
        self._xc = None
        self._xc_dot = None
        self._F_ext_prev = np.zeros(6)
        self._last_send_time = None

        self._phase = None            # 'approaching' | 'searching' | 'aligning' | 'inserting'
        self._phase_t0 = None
        self._contact_pose = None
        self._transition_vel = np.zeros(6)  # 相位切換那一刻的速度，供blend_velocity過渡用

        # 2026-08-21新增：獨立於console的0.5秒throttle log之外，另外開一份
        # 100Hz逐拍寫入的CSV，供之後拿去畫圖分析用(throttle log看趨勢夠用，
        # 但看不出真正逐拍的變化率，這個需求前面討論驟降偵測時反覆出現)。
        self._csv_file = None
        self._csv_writer = None

        self.create_subscription(WrenchStamped, 'optoforce/wrench', self._wrench_cb, 10)
        self.create_subscription(FeedbackState, 'feedback_states', self._feedback_cb, 10)

        self.set_positions_cli = self.create_client(SetPositions, 'set_positions')
        self.send_script_cli = self.create_client(SendScript, 'send_script')
        self.zero_cli = self.create_client(Trigger, 'optoforce/zero')

        self.create_service(Trigger, 'stage3_admittance/start_insertion', self._start_cb)
        self.create_service(Trigger, 'stage3_admittance/stop_insertion', self._stop_cb)

        rate = self.get_parameter('control_rate_hz').value
        self._dt = 1.0 / rate
        self.timer = self.create_timer(self._dt, self._control_loop)

        #self.get_logger().info(f"Mei node start")

    def _declare_params(self):
        p = self.declare_parameter
        p('control_rate_hz', 100.0)
        p('command_interface', 'pvt')

        # 只有Aligning/Inserting會用到這組(Approaching/Searching是剛性
        # 軌跡+獨立的Z力控制，不經過admittance_step)。
        p('admittance.selected_axes', [True, True, True, False, False, False])
        p('admittance.mass', [3.0, 3.0, 3.0, 1.0, 1.0, 1.0])
        p('admittance.stiffness', [100.0, 100.0, 250.0, 1.0, 1.0, 1.0])
        p('admittance.damping', [34.64, 34.64, 54.77, 2.0, 2.0, 2.0])

        p('insertion.insertion_depth', 0.13)
        p('insertion.insertion_speed', 0.005)
        p('insertion.completion_rate_threshold', 50.0)
        p('insertion.completion_min_travel_fraction', 0.5)
        p('insertion.early_contact_force_threshold', 8.0)

        p('insertion.contact_force_threshold', 0.4)

        # Searching階段的Z軸力控制(force_hold_z_velocity)，不是admittance
        # 的F_desired，見檔頭說明差異。
        p('insertion.search_desired_force', 1.0)     # N，希望維持的貼壓力
        p('insertion.search_z_gain', 0.005)           # m/s per N，比例增益
        p('insertion.search_z_max_speed', 0.003)      # m/s，Z力控制的速度上限

        # 2026-08-21新增：Approaching→Searching切換瞬間，手臂還帶著穩定
        # 下降速度，但Searching自然算出來的速度接近0(螺旋剛開始很慢、
        # Z軸換成獨立力控制不延續下降速度)，若沒有過渡會有瞬間速度不
        # 連續造成頓挫，用blend_velocity()在這段時間內線性混合新舊速度。
        p('insertion.phase_transition_blend_sec', 0.2)

        p('insertion.search_pitch', 0.001)
        p('insertion.search_max_radius', 0.008)
        p('insertion.search_angular_speed', 2.0)
        p('insertion.found_hole_rate_threshold', 20.0)
        p('insertion.search_timeout_sec', 30.0)
        p('insertion.search_settle_time_sec', 1.2)

        p('insertion.align_depth', 0.002)
        p('insertion.align_speed', 0.003)
        p('insertion.reject_force_threshold', 0.5)

        p('ft_sensor.filter_coefficient', 0.05)

        p('sensor_frame.r_g_s_flat', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

        p('safety.deadband_force', [0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        p('safety.max_step_correction', 0.005)
        p('safety.workspace_max_radius', 0.05)
        p('safety.workspace_repulsion_gain', 50.0)
        p('safety.force_limit', 30.0)
        p('safety.torque_limit', 3.0)
        p('safety.min_project_speed', 20)
        p('command.set_positions_velocity', 0.25)
        p('command.set_positions_acc_time', 0.05)
        p('command.min_send_interval_sec', 0.2)
        p('command.pvt_point_time_sec', 0.007)

    def _wrench_cb(self, msg: WrenchStamped):
        self._latest_wrench_raw = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
        ])

    def _feedback_cb(self, msg: FeedbackState):
        self._latest_feedback_msg = msg
        if len(msg.tool_pose) == 6:
            self._latest_tool_pose = np.array(msg.tool_pose)

    def _preflight_check(self):
        msg = self._latest_feedback_msg
        if msg is None:
            return False, '還沒收到FeedbackState，無法檢查連線狀態'
        if msg.e_stop:
            return False, '機器人處於緊急停止狀態，無法開始'
        if msg.robot_error:
            return False, f'機器人有未清除的錯誤(error_code={msg.error_code})，無法開始'
        if not msg.is_sct_connected:
            return False, 'Listen Node(SCT)未連線，請確認TMflow的Listen task正在執行'

        min_speed = self.get_parameter('safety.min_project_speed').value
        if msg.project_speed < min_speed:
            self.get_logger().warn(
                f'project_speed={msg.project_speed}%偏低(建議>={min_speed}%)，'
                f'導納反應可能因為機器人本身被限速而跟不上，仍會繼續開始'
            )
        return True, 'ok'

    def _start_cb(self, request, response):
        if self._latest_tool_pose is None:
            response.success = False
            response.message = '還沒收到feedback_states的tool_pose，無法開始'
            return response

        preflight_ok, preflight_msg = self._preflight_check()
        if not preflight_ok:
            response.success = False
            response.message = preflight_msg
            return response

        if not self.zero_cli.service_is_ready():
            response.success = False
            response.message = 'optoforce/zero service未就緒，檢查optoforce_node有沒有在跑'
            return response

        self.zero_cli.call_async(Trigger.Request())
        time.sleep(ZERO_SETTLE_SEC)
        self.get_logger().info(f'已送出optoforce/zero，等待{ZERO_SETTLE_SEC}秒讓讀值穩定')

        R_g_s = np.array(self.get_parameter('sensor_frame.r_g_s_flat').value).reshape(3, 3)
        self._R_g_s = R_g_s

        self._start_pose_base = self._latest_tool_pose.copy()
        self._R_start_to_base = euler_to_matrix(*self._start_pose_base[3:6])
        start_pos_insertion_frame = self._R_start_to_base.T @ self._start_pose_base[0:3]
        self._start_pose_insertion_frame = np.concatenate(
            [start_pos_insertion_frame, self._start_pose_base[3:6]]
        )

        self._xc = self._start_pose_insertion_frame.copy()
        self._xc_dot = np.zeros(6)
        self._transition_vel = np.zeros(6)
        self._F_ext_prev = np.zeros(6)
        self._last_send_time = None
        self._t0 = self.get_clock().now()
        self._phase = 'approaching'
        self._phase_t0 = self._t0
        self._contact_pose = None
        self._workspace_center = self._start_pose_base[0:3].copy()
        self._active = True

        csv_path = f'rigid_search_data_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        self._csv_file = open(csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            't_sec', 'phase', 'F_ext_x', 'F_ext_y', 'F_ext_z',
            'correction_x_mm', 'correction_y_mm', 'correction_z_mm',
            'tool_rot_x_deg', 'tool_rot_y_deg', 'tool_rot_z_deg',
        ])
        self.get_logger().info(f'逐拍(100Hz)資料記錄到 {csv_path}')

        if self.get_parameter('command_interface').value == 'pvt':
            self._call_send_script('PvtEnter', 'PVTEnter(1)')

        response.success = True
        response.message = '已歸零，開始剛性搜尋插入'
        return response

    def _stop_cb(self, request, response):
        self._deactivate('手動停止', hard_stop=True)
        response.success = True
        response.message = '已停止'
        return response

    def _deactivate(self, reason: str, hard_stop: bool = False):
        if self._active and hard_stop:
            self._call_send_script('StopBuf', 'StopAndClearBuffer(0)')
        if self._active and self.get_parameter('command_interface').value == 'pvt':
            self._call_send_script('PvtExit', 'PVTExit()')
        self._active = False
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        self.get_logger().info(f'剛性搜尋插入停止：{reason}')

    def _control_loop(self):
        if not self._active:
            return
        if self._latest_wrench_raw is None or self._latest_tool_pose is None:
            return

        F_ext = self._latest_wrench_raw.copy()

        R_sensor_to_insertion = self._R_g_s
        F_ext[0:3] = R_sensor_to_insertion @ F_ext[0:3]
        F_ext[3:6] = R_sensor_to_insertion @ F_ext[3:6]

        F_z_prev_filtered = self._F_ext_prev[2]

        alpha = self.get_parameter('ft_sensor.filter_coefficient').value
        F_ext = filter_wrench(F_ext, self._F_ext_prev, alpha)
        self._F_ext_prev = F_ext
        F_z_current_filtered = F_ext[2]

        deadband = np.array(self.get_parameter('safety.deadband_force').value)
        F_ext = apply_deadband(F_ext, deadband)

        force_limit = self.get_parameter('safety.force_limit').value
        torque_limit = self.get_parameter('safety.torque_limit').value
        if not check_force_limit(F_ext, force_limit, torque_limit):
            self._deactivate(f'力/力矩超過安全閾值 F_ext={F_ext}', hard_stop=True)
            return

        t_phase = (self.get_clock().now() - self._phase_t0).nanoseconds * 1e-9

        if self._phase == 'approaching':
            # 2026-08-21修正：原本X/Y/Z全部剛性(xc=xd)，實機測試在硬桌面
            # 上碰撞力量會直接衝到2N以上，因為碰到東西的瞬間到偵測到閾值
            # 之間那一拍(100Hz，0.01秒)軌跡還是硬邦邦往下走，沒有任何緩衝
            # 空間。曾經改成跟insertion_node_zero.py一樣只鎖X/Y、Z軸開放
            # 導納，但使用者要求先註解掉、退回完全剛性，原因待補充——
            # 保留這段程式碼在註解裡，之後要重新啟用直接取消註解即可。
            # M = np.array(self.get_parameter('admittance.mass').value)
            # K = np.array(self.get_parameter('admittance.stiffness').value)
            # D = np.array(self.get_parameter('admittance.damping').value)
            # selected_axes = np.array(self.get_parameter('admittance.selected_axes').value)
            # selected_axes_effective = selected_axes.copy()
            # selected_axes_effective[0:2] = False
            #
            # xc_new, xc_dot_new = admittance_step(
            #     F_ext, self._xc, self._xc_dot, xd, xd_dot, xd_ddot, M, K, D, self._dt, selected_axes_effective
            # )
            depth = self.get_parameter('insertion.insertion_depth').value
            speed = self.get_parameter('insertion.insertion_speed').value
            xd, xd_dot, xd_ddot = linear_insertion_trajectory(
                t_phase, self._start_pose_insertion_frame, depth, speed
            )
            xc_new = xd.copy()
            xc_dot_new = xd_dot.copy()

        elif self._phase == 'searching':
            # X/Y/旋轉軸剛性跟著螺旋軌跡走(penetration_depth=0，Z由下面
            # 的force_hold_z_velocity獨立控制，不用spiral_trajectory算
            # 出來的Z)。
            pitch = self.get_parameter('insertion.search_pitch').value
            search_max_radius = self.get_parameter('insertion.search_max_radius').value
            angular_speed = self.get_parameter('insertion.search_angular_speed').value
            xd, xd_dot, xd_ddot = spiral_trajectory(
                t_phase, self._contact_pose, 0.0, pitch, search_max_radius, angular_speed
            )

            search_force = self.get_parameter('insertion.search_desired_force').value
            gain = self.get_parameter('insertion.search_z_gain').value
            max_speed = self.get_parameter('insertion.search_z_max_speed').value
            # 2026-08-20實機測試發現bug並修正：這個專案的慣例是碰觸表面
            # 時F_ext[2]讀值為負(Approaching的碰觸判定就是拿F_ext[2]跟
            # 負值比較，log實測Fz=-0.41N才觸發)，但force_hold_z_velocity
            # 的設計是標準正值慣例(actual<desired就往下壓，兩者都預期是
            # 正值)。原本直接把F_ext[2](負值)傳進去，導致力量真的增加、
            # 讀值變更負時，誤差項反而越算越大，控制器誤判成"還不夠力"、
            # 持續下壓不停，直到撞上force_limit安全停止。這裡把F_ext[2]
            # 先取負號轉成正值的"貼壓力大小"再傳進去，才能對齊函式內部
            # 標準正值慣例的假設，search_desired_force本身不用改符號。
            z_vel = force_hold_z_velocity(-F_ext[2], search_force, gain, max_speed)

            xc_new = xd.copy()
            xc_new[2] = self._xc[2] + z_vel * self._dt
            xc_dot_new = xd_dot.copy()
            xc_dot_new[2] = z_vel

            # 2026-08-21新增：跟Approaching切換那一刻的速度做線性過渡，
            # 避免瞬間速度不連續造成頓挫，見phase_transition_blend_sec
            # 宣告處的說明。只影響送給機器人的速度指令，不影響xc_new
            # 位置本身(位置還是照自然算出來的走)。
            blend_sec = self.get_parameter('insertion.phase_transition_blend_sec').value
            xc_dot_new = blend_velocity(self._transition_vel, xc_dot_new, t_phase, blend_sec)

        elif self._phase == 'aligning':
            # Step3起改用真正的導納(admittance_step)，理由見檔頭說明。
            M = np.array(self.get_parameter('admittance.mass').value)
            K = np.array(self.get_parameter('admittance.stiffness').value)
            D = np.array(self.get_parameter('admittance.damping').value)
            selected_axes = np.array(self.get_parameter('admittance.selected_axes').value)

            align_depth = self.get_parameter('insertion.align_depth').value
            align_speed = self.get_parameter('insertion.align_speed').value
            xd, xd_dot, xd_ddot, step3_done = aligning_trajectory(
                t_phase, self._contact_pose, align_depth, align_speed
            )
            xc_new, xc_dot_new = admittance_step(
                F_ext, self._xc, self._xc_dot, xd, xd_dot, xd_ddot, M, K, D, self._dt, selected_axes
            )
            # 2026-08-21新增：跟Searching切換過來那一刻的速度做線性過渡，
            # 理由跟Approaching→Searching那次一樣(避免瞬間速度不連續造成
            # 頓挫)，只影響送給機器人的速度指令，self._xc_dot本身還是從0
            # 開始積分(admittance_step的輸入不變)，維持臨界阻尼不超調的
            # 特性，只有"要送出去的目標速度"做平滑。
            blend_sec = self.get_parameter('insertion.phase_transition_blend_sec').value
            xc_dot_new = blend_velocity(self._transition_vel, xc_dot_new, t_phase, blend_sec)

        elif self._phase == 'inserting':
            M = np.array(self.get_parameter('admittance.mass').value)
            K = np.array(self.get_parameter('admittance.stiffness').value)
            D = np.array(self.get_parameter('admittance.damping').value)
            selected_axes = np.array(self.get_parameter('admittance.selected_axes').value)

            remaining_depth = self.get_parameter('insertion.insertion_depth').value \
                - self.get_parameter('insertion.align_depth').value
            speed = self.get_parameter('insertion.insertion_speed').value
            xd, xd_dot, xd_ddot = linear_insertion_trajectory(
                t_phase, self._contact_pose, remaining_depth, speed
            )
            xc_new, xc_dot_new = admittance_step(
                F_ext, self._xc, self._xc_dot, xd, xd_dot, xd_ddot, M, K, D, self._dt, selected_axes
            )
            # 2026-08-21新增：跟Aligning切換過來那一刻的速度做線性過渡，
            # 理由同上。
            blend_sec = self.get_parameter('insertion.phase_transition_blend_sec').value
            xc_dot_new = blend_velocity(self._transition_vel, xc_dot_new, t_phase, blend_sec)

        else:
            self.get_logger().error(f'未知的_phase={self._phase!r}，停止')
            self._deactivate('內部狀態錯誤：未知的phase', hard_stop=True)
            return

        max_step = self.get_parameter('safety.max_step_correction').value
        correction = xc_new[0:3] - xd[0:3]
        correction = saturate_vector(correction, max_step)
        xc_new[0:3] = xd[0:3] + correction

        tool_rot_deg = np.degrees(self._latest_tool_pose[3:6])
        self.get_logger().info(
            f'[{self._phase}] F_ext(xyz)=[{F_ext[0]:.2f},{F_ext[1]:.2f},{F_ext[2]:.2f}]N  '
            f'correction(xyz)=[{correction[0]*1000:.2f},{correction[1]*1000:.2f},{correction[2]*1000:.2f}]mm  '
            f'tool_rot(deg)=[{tool_rot_deg[0]:.2f},{tool_rot_deg[1]:.2f},{tool_rot_deg[2]:.2f}]',
            throttle_duration_sec=0.5,
        )

        # 2026-08-21新增：跟上面console的throttle log分開，這裡每一拍
        # (100Hz)都寫一筆到CSV，不受throttle影響，供之後畫圖分析逐拍
        # 變化用(throttle log只適合看趨勢，看不出真正的逐拍變化率)。
        if self._csv_writer is not None:
            t_total = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
            self._csv_writer.writerow([
                f'{t_total:.4f}', self._phase,
                f'{F_ext[0]:.4f}', f'{F_ext[1]:.4f}', f'{F_ext[2]:.4f}',
                f'{correction[0]*1000:.4f}', f'{correction[1]*1000:.4f}', f'{correction[2]*1000:.4f}',
                f'{tool_rot_deg[0]:.4f}', f'{tool_rot_deg[1]:.4f}', f'{tool_rot_deg[2]:.4f}',
            ])

        pos_base = self._R_start_to_base @ xc_new[0:3]
        max_radius = self.get_parameter('safety.workspace_max_radius').value
        gain = self.get_parameter('safety.workspace_repulsion_gain').value
        repulsion_base = workspace_repulsion(pos_base, self._workspace_center, max_radius, gain)
        xc_new[0:3] += self._R_start_to_base.T @ repulsion_base

        self._xc = xc_new
        self._xc_dot = xc_dot_new

        target_pos_base = self._R_start_to_base @ xc_new[0:3]
        target_pose_base = np.concatenate([target_pos_base, self._start_pose_base[3:6]])
        target_vel_base = np.concatenate([self._R_start_to_base @ xc_dot_new[0:3], np.zeros(3)])

        if self._phase == 'approaching':
            depth = self.get_parameter('insertion.insertion_depth').value
            speed = self.get_parameter('insertion.insertion_speed').value
            contact_threshold = self.get_parameter('insertion.contact_force_threshold').value
            z_travelled = min(speed * t_phase, depth)

            if abs(F_ext[2]) > contact_threshold:
                self._send_target(target_pose_base, target_vel_base)
                self._contact_pose = xd.copy()
                self._xc = self._contact_pose.copy()
                # 2026-08-21：這裡改記錄切換那一刻的實際速度(xc_dot_new，
                # 剛才已經存進self._xc_dot)，供Searching開頭用blend_velocity
                # 做平滑過渡，不再直接歸零(歸零會導致下一階段的目標速度
                # 瞬間從"穩定下降"變成"接近0"，造成頓挫，見phase_transition_
                # blend_sec宣告處的說明)。
                self._transition_vel = self._xc_dot.copy()
                self._xc_dot = np.zeros(6)
                self._phase = 'searching'
                self._phase_t0 = self.get_clock().now()
                self.get_logger().info(
                    f'Approaching：偵測到碰觸(Fz={F_ext[2]:.2f}N，閾值{contact_threshold}N)，'
                    '切換到Searching階段(剛性螺旋搜尋+Z力控制)'
                )
                return
            elif z_travelled >= depth:
                self._send_target(target_pose_base, target_vel_base)
                self._deactivate(
                    f'Approaching：已下降完insertion_depth({depth*1000:.1f}mm)仍未偵測到碰觸，'
                    f'可能沒對準孔位或contact_force_threshold設太高，需人工確認'
                )
                return

        elif self._phase == 'searching':
            found_rate_threshold = self.get_parameter('insertion.found_hole_rate_threshold').value
            timeout = self.get_parameter('insertion.search_timeout_sec').value
            settle_time = self.get_parameter('insertion.search_settle_time_sec').value
            found = (
                t_phase > settle_time
                and detect_force_rate_drop(F_z_current_filtered, F_z_prev_filtered, self._dt, found_rate_threshold)
            )

            if found:
                self._send_target(target_pose_base, target_vel_base)
                self._contact_pose = xc_new.copy()
                self._xc = self._contact_pose.copy()
                self._transition_vel = self._xc_dot.copy()
                self._xc_dot = np.zeros(6)
                self._phase = 'aligning'
                self._phase_t0 = self.get_clock().now()
                self.get_logger().info(
                    'Searching：偵測到Fz驟降(疑似找到孔)，切換到Aligning階段(對準滑入，開始使用導納)'
                )
                return
            elif t_phase >= timeout:
                self._send_target(target_pose_base, target_vel_base)
                self._deactivate(
                    f'Searching：螺旋搜尋超過{timeout}秒仍未找到孔(搜尋半徑已達'
                    f'{self.get_parameter("insertion.search_max_radius").value*1000:.1f}mm上限)，'
                    '搜尋失敗，需人工重新定位後再開始'
                )
                return

        elif self._phase == 'aligning':
            reject_threshold = self.get_parameter('insertion.reject_force_threshold').value
            if check_step3_false_positive(F_z_current_filtered, reject_threshold):
                self._send_target(target_pose_base, target_vel_base)
                self._deactivate(
                    f'Aligning：延續下降時力量又超過{reject_threshold}N，判定Step2的驟降'
                    '是誤判(沒有真的滑進孔)，需人工重新定位後再開始'
                )
                return
            elif step3_done:
                self._send_target(target_pose_base, target_vel_base)
                self._contact_pose = xd.copy()
                self._transition_vel = self._xc_dot.copy()
                self._xc_dot = np.zeros(6)
                self._phase = 'inserting'
                self._phase_t0 = self.get_clock().now()
                self.get_logger().info('Aligning：對準滑入完成，切換到Inserting階段(插入到底)')
                return

        elif self._phase == 'inserting':
            remaining_depth = self.get_parameter('insertion.insertion_depth').value \
                - self.get_parameter('insertion.align_depth').value
            speed = self.get_parameter('insertion.insertion_speed').value
            completion_rate_threshold = self.get_parameter('insertion.completion_rate_threshold').value
            completion_min_travel_fraction = self.get_parameter('insertion.completion_min_travel_fraction').value
            z_travelled = min(speed * t_phase, remaining_depth)
            travel_fraction = z_travelled / remaining_depth if remaining_depth > 1e-9 else 1.0

            completed = (
                travel_fraction >= completion_min_travel_fraction
                and detect_force_rate_rise(F_z_current_filtered, F_z_prev_filtered, self._dt, completion_rate_threshold)
            )

            if completed:
                self._send_target(target_pose_base, target_vel_base)
                self._deactivate('Inserting：偵測到Fz驟升(插到底了)，插入完成')
                return
            elif z_travelled >= remaining_depth:
                self._send_target(target_pose_base, target_vel_base)
                self._deactivate(
                    f'Inserting：已走完剩餘深度({remaining_depth*1000:.1f}mm)仍未偵測到驟升，'
                    '可能沒插到底或completion_rate_threshold設太高，需人工確認'
                )
                return

        self._send_target(target_pose_base)

    def _send_target(self, pose_base, vel_base=None):
        if self.get_parameter('command_interface').value == 'pvt':
            self._send_via_pvt(pose_base, vel_base)
        else:
            self._send_via_set_positions(pose_base)

    def _send_via_set_positions(self, pose_base):
        if not self.set_positions_cli.service_is_ready():
            self.get_logger().warn('set_positions service未就緒，這一拍沒有送出指令', throttle_duration_sec=1.0)
            return
        min_interval = self.get_parameter('command.min_send_interval_sec').value
        now = self.get_clock().now()
        if self._last_send_time is not None:
            elapsed = (now - self._last_send_time).nanoseconds * 1e-9
            if elapsed < min_interval:
                return
        self._last_send_time = now
        req = SetPositions.Request()
        req.motion_type = SetPositions.Request.LINE_T
        req.positions = pose_base.tolist()
        req.velocity = self.get_parameter('command.set_positions_velocity').value
        req.acc_time = self.get_parameter('command.set_positions_acc_time').value
        req.blend_percentage = 0
        req.fine_goal = False
        future = self.set_positions_cli.call_async(req)
        future.add_done_callback(self._check_set_positions_result)

    def _check_set_positions_result(self, future):
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warn(f'SetPositions呼叫例外：{exc}', throttle_duration_sec=1.0)
            return
        if resp is not None and not resp.ok:
            self.get_logger().warn('SetPositions回應ok=False，本地端TCP送出失敗(逾時)', throttle_duration_sec=1.0)

    def _send_via_pvt(self, pose_base, vel_base=None):
        if not self.send_script_cli.service_is_ready():
            self.get_logger().warn('send_script service未就緒，這一拍沒有送出PVTPoint', throttle_duration_sec=1.0)
            return
        elapsed = self.get_parameter('command.pvt_point_time_sec').value

        pos_mm = pose_base[0:3] * 1000.0
        rot_deg = np.degrees(pose_base[3:6])
        pos_str = ",".join(f"{v:.5f}" for v in np.concatenate([pos_mm, rot_deg]))
        if vel_base is not None:
            vel_mm_s = vel_base[0:3] * 1000.0
            vel_deg_s = np.degrees(vel_base[3:6])
            vel_str = ",".join(f"{v:.5f}" for v in np.concatenate([vel_mm_s, vel_deg_s]))
        else:
            vel_str = ",".join(["0.0"] * 6)
        script = f"PVTPoint({pos_str},{vel_str},{elapsed:.4f})"
        self._call_send_script('PvtPt', script, check_result=True)

    def _call_send_script(self, id_str, script, check_result=False):
        if not self.send_script_cli.service_is_ready():
            return
        req = SendScript.Request()
        req.id = id_str
        req.script = script
        future = self.send_script_cli.call_async(req)
        if check_result:
            future.add_done_callback(self._check_send_script_result)

    def _check_send_script_result(self, future):
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warn(f'SendScript(PVTPoint)呼叫例外：{exc}', throttle_duration_sec=1.0)
            return
        if resp is not None and not resp.ok:
            self.get_logger().warn('SendScript(PVTPoint)回應ok=False，本地端TCP送出失敗(逾時)', throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = InsertionNodeRigidSearch()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
