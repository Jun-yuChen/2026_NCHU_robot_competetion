import rclpy
from rclpy.node import Node
import time

from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SetPositions, SetIO

# ============================================================
# 【主機插孔固定座標，跟 arm_goto_fixed_table.py 同一套做法】
# 差異：這裡的「抓取點」是接頭目前插在主機孔位上的位置，抓緊後沿 -X
# 退出的動作就是實際「拔出」的動作，退出後跟原本一樣回到主機前方。
# 單位：x/y/z 是公尺 (m)，rx/ry/rz 是弧度 (rad)。
#
# 執行時輸入 1 / 2 就會直接飛到對應的插孔座標。
# ============================================================
UNPLUG_POSES_M = {
    # 原始量測 (robot base)：x=640.56, y=-117.49, z=368.71 (mm)
    #                        rx=93.47, ry=3.16, rz=92.04 (deg)
    1: [0.64056, -0.11749, 0.36871, 1.631359, 0.055152, 1.606401],

    # 原始量測 (robot base)：x=641.24, y=-128.66, z=359.80 (mm)
    #                        rx=96.35, ry=0.45, rz=94.86 (deg)
    2: [0.64124, -0.12866, 0.35980, 1.681625, 0.007854, 1.655619],
}

ARRIVE_THRESHOLD_M = 0.005      # 判定「已抵達」的距離誤差 (m)

# 【1、2 號插孔各自的預備位置】：姿態跟對應插孔完全相同，只有 X 往後退 150mm，
# 先到這裡打開夾爪，再從這裡移動到插孔位置夾住接頭。
APPROACH_POSES_M = {
    # x = 640.56 - 150 = 490.56 (mm)，其餘同 1 號插孔
    1: [0.49056, -0.11749, 0.36871, 1.631359, 0.055152, 1.606401],

    # x = 641.24 - 150 = 491.24 (mm)，其餘同 2 號插孔
    2: [0.49124, -0.12866, 0.35980, 1.681625, 0.007854, 1.655619],
}

# 夾住接頭後，沿底座 -X 方向直線退出這個距離——這個動作本身就是「拔出」，
# 退出後才轉向主機前方，避免夾著接頭斜向拉扯把插孔或接頭扯壞。
RETREAT_DISTANCE_M = 0.15

# 【主機前方固定放置點】：拔出後固定回到這裡，跟 arm_goto_fixed_table.py 相同。
# 原始量測 (robot base)：x=371.82, y=-155.44, z=423.01 (mm)
#                        rx=93.88, ry=0.89, rz=88.02 (deg)
HOST_FRONT_POSE_M = [0.37182, -0.15544, 0.42301, 1.638515, 0.015533, 1.536239]

# 【夾爪 IO 設定，沿用 arm_cmd_simple.py 的 module=1/type=1/pin=0 慣例】
GRIPPER_OPEN_STATE = 0.0
GRIPPER_CLOSE_STATE = 1.0
GRIPPER_ACTUATION_DELAY_S = 1.0   # 送出開合指令後，等夾爪實際動作完成的緩衝時間

# 【移動速度，覺得太慢/太快都直接調這兩組就好】
TRANSIT_VELOCITY = 0.35    # LINE_T 長距離移動 (預備點、回主機前方)——原本用 PTP_T 會繞大圈，改走卡氏直線
TRANSIT_ACC_TIME = 0.25
PRECISION_VELOCITY = 0.15  # LINE_T 短距離精準移動 (預備點→插孔、拔出)
PRECISION_ACC_TIME = 0.25


class ArmUnplugHostPorts(Node):
    def __init__(self):
        super().__init__('arm_unplug_host_ports')
        self.current_flange_pose_m = None

        self.sub_feedback = self.create_subscription(FeedbackState, 'feedback_states', self.feedback_callback, 10)
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
        if len(raw_pose) < 3:
            return
        p0, p1, p2 = raw_pose[0], raw_pose[1], raw_pose[2]
        # TM 驅動有時回傳 mm、有時回傳 m，用數值大小粗略判斷單位並統一換算成 m
        if abs(p0) > 5.0 or abs(p1) > 5.0 or abs(p2) > 5.0:
            self.current_flange_pose_m = [float(p0 / 1000.0), float(p1 / 1000.0), float(p2 / 1000.0)]
        else:
            self.current_flange_pose_m = [float(p0), float(p1), float(p2)]

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
                "流程指標停在 Listen 節點、Auto 模式、或 Safeguard 是否被觸發。"
            )
            return False
        return True

    def set_gripper(self, state):
        """state = 0.0 打開夾爪, state = 1.0 閉合夾爪抓取"""
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

    def wait_until_arrived(self, target_pos_m, threshold_m=ARRIVE_THRESHOLD_M):
        # 不設逾時，會一直等到真的進入門檻內為止；要中止只能 Ctrl+C。
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.current_flange_pose_m is not None:
                dist = sum((self.current_flange_pose_m[i] - target_pos_m[i]) ** 2 for i in range(3)) ** 0.5
                if dist <= threshold_m:
                    return True
        return False


def main(args=None):
    rclpy.init(args=args)
    node = ArmUnplugHostPorts()

    # 等第一筆手臂回饋進來，確保 current_flange_pose_m 有值
    while rclpy.ok() and node.current_flange_pose_m is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    host_pos, host_rpy = HOST_FRONT_POSE_M[:3], HOST_FRONT_POSE_M[3:]
    valid_keys = ", ".join(str(k) for k in sorted(UNPLUG_POSES_M.keys()))
    node.get_logger().info(f"已就緒，輸入 {valid_keys} 選擇要拔出的主機插孔，輸入 q 離開。")

    try:
        while rclpy.ok():
            try:
                choice = input(f"移動到插孔編號 ({valid_keys}，q 離開): ").strip()
            except EOFError:
                break

            if choice.lower() == 'q':
                break

            if not choice.isdigit() or int(choice) not in UNPLUG_POSES_M:
                print(f"請輸入 {valid_keys} 其中一個數字，或輸入 q 離開。")
                continue

            key = int(choice)
            pose = UNPLUG_POSES_M[key]
            pos, rpy = pose[:3], pose[3:]

            approach = APPROACH_POSES_M.get(key)
            if approach is not None:
                ap_pos, ap_rpy = approach[:3], approach[3:]
                node.get_logger().info(f"先移動到 {key} 號插孔預備位置: {approach}")
                ok = node.send_motion(ap_pos, ap_rpy, velocity_val=TRANSIT_VELOCITY, acc_time_val=TRANSIT_ACC_TIME,
                                       motion_type=SetPositions.Request.LINE_T)
                if ok:
                    ok = node.wait_until_arrived(ap_pos)
                if not ok:
                    node.get_logger().error(f">>> 未能抵達 {key} 號插孔預備位置，取消拔出，請檢查上面的錯誤訊息！ <<<")
                    continue
                node.set_gripper(GRIPPER_OPEN_STATE)
                time.sleep(GRIPPER_ACTUATION_DELAY_S)

            node.get_logger().info(f"移動到插孔 {key}: {pose}")
            ok = node.send_motion(pos, rpy, velocity_val=PRECISION_VELOCITY, acc_time_val=PRECISION_ACC_TIME,
                                   motion_type=SetPositions.Request.LINE_T)
            if ok:
                ok = node.wait_until_arrived(pos)

            if not ok:
                node.get_logger().error(f">>> 未能抵達插孔 {key}，取消拔出，請檢查上面的錯誤訊息！ <<<")
                continue

            node.get_logger().info(f">>> 已抵達插孔 {key}，夾爪夾住接頭中... <<<")
            node.set_gripper(GRIPPER_CLOSE_STATE)
            time.sleep(GRIPPER_ACTUATION_DELAY_S)

            retreat_pos = [pos[0] - RETREAT_DISTANCE_M, pos[1], pos[2]]
            node.get_logger().info(f"夾住接頭，沿 -X 拔出 {RETREAT_DISTANCE_M * 1000:.0f}mm: {retreat_pos}")
            retreat_ok = node.send_motion(retreat_pos, rpy, velocity_val=PRECISION_VELOCITY, acc_time_val=PRECISION_ACC_TIME,
                                           motion_type=SetPositions.Request.LINE_T)
            if retreat_ok:
                retreat_ok = node.wait_until_arrived(retreat_pos)

            if not retreat_ok:
                node.get_logger().error(f">>> 拔出失敗，取消返回主機前方，請檢查上面的錯誤訊息！ <<<")
                continue

            node.get_logger().info(f"返回主機前方: {HOST_FRONT_POSE_M}")
            back_ok = node.send_motion(host_pos, host_rpy, velocity_val=TRANSIT_VELOCITY, acc_time_val=TRANSIT_ACC_TIME,
                                        motion_type=SetPositions.Request.LINE_T)
            if back_ok:
                back_ok = node.wait_until_arrived(host_pos)

            if back_ok:
                node.get_logger().info(">>> 已回到主機前方！ <<<")
            else:
                node.get_logger().error(">>> 未能回到主機前方，請檢查上面的錯誤訊息！ <<<")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
