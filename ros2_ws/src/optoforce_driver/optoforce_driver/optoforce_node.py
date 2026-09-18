"""OptoForce HEX-70-CE-2000N 六軸力/扭矩感測器的ROS2驅動節點。

沒有官方ROS/ROS2套件，直接照廠商協定文件自己解封包(桌面
OPTOFORCE/DAQ material/USB/_Protocol description/ 那份PDF，
"OptoForce General DAQ - protocol description 1.4.1")：

USB走CDC虛擬序列埠(1,000,000 baud, 8N1, 無流控)，開機後預設100Hz
持續傳送22 bytes的資料封包，全部欄位big-endian(高位元組在前)：

    Header(170,7,8,16) 4bytes + SampleCounter(UINT16) + Status(UINT16)
    + Fx,Fy,Fz,Tx,Ty,Tz(各INT16) + Checksum(UINT16)

checksum = 前面20個bytes的總和(取UINT16)。

counts轉N/Nm的係數是這顆感測器(序號ICE042)出廠校正報告
(Sensor datasheet/SensitivityReport_ICE0A034.pdf)量出來的專屬值，
換一顆感測器(或重新校正)就要換這幾個數字，不是通用常數。

⚠️ 這台是USB-CDC虛擬序列埠，只用bulk/interrupt傳輸，不像RealSense
   相機需要isochronous傳輸——WSL2透過usbipd-win passthrough應該可以
   正常使用，不會踩到相機那個「WSL2不支援isochronous」的坑。

用法：
    ros2 run optoforce_driver optoforce_node --ros-args -p port:=/dev/ttyACM0

歸零介面(給其他node呼叫)：
    std_srvs/Trigger service，其他node呼叫 optoforce/zero，這通call會
    block到offset重新收集真的完成(或逾時)才回傳，response.success +
    response.message("SENSOR_OK"/"SENSOR_TIMEOUT: ...")。沒有用action，
    因為這個操作不需要cancel或進度feedback，單純blocking call/response
    就夠——但底層還是需要跟action版一樣的thread-safe handoff，因為
    「送出歸零指令」跟「offset重新收集完成」中間隔了offset_samples筆
    封包的時間，service callback不能用忙等或跨thread直接讀旗標的方式
    等它完成。

    Thread model: _poll()是唯一會寫_history/_offset/_offset_buffer/
    _offset_ready的地方(timer callback thread，MutuallyExclusiveCallbackGroup)。
    service的_zero_callback(ReentrantCallbackGroup，跑在MultiThreadedExecutor
    底下的另一個thread)不直接碰這些欄位，只透過兩個thread-safe的Queue
    跟_poll溝通：
      - _zero_request_q: _zero_callback丟"該歸零了"進去，_poll在下一次
        poll開頭撈出來才真的送歸零指令+重置狀態(重置動作留在寫者自己
        的thread裡做，不跨thread碰共享狀態)。
      - _zero_done_q: _accumulate_offset收滿新offset後，如果這次重新收集
        是由service觸發的(_zero_pending_ack)，才把樣本數丟進這個queue。
        _zero_callback對它做get(timeout=...)——這個block是安全的，因為
        _poll在另一個thread繼續跑，不是忙等同一個thread的旗標。逾時就
        回傳SENSOR_TIMEOUT。
    node開機時的第一次歸零(__init__裡呼叫)不會經過_zero_pending_ack，
    所以不會誤觸發_zero_done_q，也不會留殘留樣本讓下一次真正的service
    請求撈到舊的完成信號。

    注意：service callback會被佔用最多zero_timeout_sec秒(預設同時只能
    處理一個歸零請求)，MultiThreadedExecutor的thread數要夠(至少2個)，
    否則等待期間會排擠其他callback(包括_poll，如果group沒分開的話)。
"""
import queue
import struct
import time
from collections import deque

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger

import serial

HEADER = bytes([170, 7, 8, 16])
PACKET_LEN = 22

# 序號ICE042的出廠校正係數(counts per N 或 counts per Nm)，來自
# SensitivityReport_ICE0A034.pdf。換一顆感測器要換這幾個數字。
SENSITIVITY = {
    'fx': 34.52, 'fy': 37.87, 'fz': 4.09,
    'tx': 634.50, 'ty': 603.51, 'tz': 898.24,
}


class OptoForceNode(Node):
    def __init__(self):
        super().__init__('optoforce_node')

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('frame_id', 'optoforce_sensor')
        self.declare_parameter('filter_window', 10)
        self.declare_parameter('offset_samples', 100)
        self.declare_parameter('zero_timeout_sec', 5.0)

        port = self.get_parameter('port').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.zero_timeout_sec = self.get_parameter(
            'zero_timeout_sec').get_parameter_value().double_value

        window = self.get_parameter('filter_window').get_parameter_value().integer_value
        if window < 1:
            self.get_logger().warn(f'filter_window={window}無效，改用1(不濾波)')
            window = 1
        self.filter_window = window
        self._axes = ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')
        # 每軸各自一個deque，maxlen=window，超出視窗的舊樣本會自動被擠掉。
        self._history = {axis: deque(maxlen=self.filter_window) for axis in self._axes}

        # 軟體offset：硬體歸零之外再加一層，開機後收前N筆原始讀值取平均
        # 當offset，之後每筆讀值都先減掉這個offset再進moving average濾波。
        offset_samples = self.get_parameter('offset_samples').get_parameter_value().integer_value
        self.offset_samples = max(offset_samples, 0)
        self._offset = {axis: 0.0 for axis in self._axes}
        self._offset_buffer = {axis: [] for axis in self._axes}
        self._offset_ready = (self.offset_samples == 0)

        # --- 歸零介面用的thread-safe handoff ---
        # _poll()(timer thread)是唯一寫入者；action的execute_callback
        # (另一個thread)只丟request/等done，不直接碰上面那些欄位。
        self._zero_request_q = queue.Queue()
        self._zero_done_q = queue.Queue()
        # 只有「這次的offset重新收集是action觸發的」才需要通知done queue，
        # 避免開機那次歸零也誤觸發它，或把上一次的殘留完成信號被下一次
        # 請求撈走。這個旗標只在_poll thread裡讀寫。
        self._zero_pending_ack = False

        self.get_logger().info(f'開啟序列埠 {port} (1000000 baud)...')
        self.ser = serial.Serial(port, baudrate=1_000_000, timeout=0)

        # ⚠️ 感測器的歸零設定斷電就會重置，這支driver每次開機都是回到
        # 原始未歸零基準——開機時自動送一次硬體歸零。
        time.sleep(0.2)  # 給序列埠一點時間穩定，避免第一個bytes遺失
        self._send_zero_command()

        self.pub = self.create_publisher(WrenchStamped, 'optoforce/wrench', 10)
        self._buffer = bytearray()

        # timer跟service分屬不同callback group，且main()用MultiThreadedExecutor，
        # 讓_zero_callback可以在等_zero_done_q時，_poll仍能繼續在另一個
        # thread跑，不會互相卡住。
        self._poll_group = MutuallyExclusiveCallbackGroup()
        self._zero_group = ReentrantCallbackGroup()

        self.zero_srv = self.create_service(
            Trigger, 'optoforce/zero', self._zero_callback,
            callback_group=self._zero_group,
        )

        # 感測器預設100Hz持續傳送，用200Hz輪詢確保不漏包。
        self.timer = self.create_timer(
            1.0 / 200.0, self._poll, callback_group=self._poll_group)

    def _send_zero_command(self, speed: int = 10, filt: int = 4):
        """送9-byte CONFIGURATION封包觸發硬體歸零(ZERO=255)，Speed=10
        (100Hz)、Filter=4(15Hz cutoff)維持出廠預設值不變。
        Checksum = 170+0+50+3+Speed+Filter+Zero (取UINT16)。
        """
        time.sleep(0.5)
        zero = 255
        checksum = (170 + 0 + 50 + 3 + speed + filt + zero) & 0xFFFF
        packet = bytes([170, 0, 50, 3, speed, filt, zero]) + struct.pack('>H', checksum)
        self.ser.write(packet)
        self.get_logger().info('已送出歸零指令')

    def _drain_zero_requests(self):
        """在_poll開頭呼叫。只在timer thread裡執行，所以底下這些狀態
        重置動作不需要lock。這是_zero_request_q唯一的消費端。"""
        try:
            self._zero_request_q.get_nowait()
        except queue.Empty:
            return

        self._send_zero_command()
        # 歸零基準改變了，濾波視窗跟軟體offset裡殘留的舊樣本都是用舊
        # 基準算出來的，一律清空重新累積。
        for h in self._history.values():
            h.clear()
        for axis in self._axes:
            self._offset[axis] = 0.0
            self._offset_buffer[axis].clear()
        self._offset_ready = (self.offset_samples == 0)
        self._zero_pending_ack = True

        if self._offset_ready:
            # offset_samples設成0(關閉軟體offset)，沒有樣本要收集，
            # 立刻視為完成。
            self._zero_pending_ack = False
            self._zero_done_q.put(0)

    def _zero_callback(self, request, response):
        # 先清掉done queue裡任何殘留項目(理論上不該有，但避免萬一撈到
        # 不屬於這次請求的舊完成信號)。
        while True:
            try:
                self._zero_done_q.get_nowait()
            except queue.Empty:
                break

        self._zero_request_q.put(True)

        try:
            samples = self._zero_done_q.get(timeout=self.zero_timeout_sec)
        except queue.Empty:
            response.success = False
            response.message = (
                f'SENSOR_TIMEOUT: 歸零指令已送出，但offset重新收集在'
                f'{self.zero_timeout_sec:.1f}秒內未完成')
            return response

        response.success = True
        response.message = f'SENSOR_OK (samples_collected={samples})'
        return response

    def _poll(self):
        self._drain_zero_requests()

        n = self.ser.in_waiting
        if n:
            self._buffer += self.ser.read(n)

        while True:
            idx = self._buffer.find(HEADER)
            if idx < 0:
                if len(self._buffer) > 3:
                    del self._buffer[:-3]
                return
            if idx > 0:
                del self._buffer[:idx]
            if len(self._buffer) < PACKET_LEN:
                return

            packet = bytes(self._buffer[:PACKET_LEN])
            del self._buffer[:PACKET_LEN]
            self._parse_and_publish(packet)

    def _parse_and_publish(self, packet: bytes):
        checksum_calc = sum(packet[:20]) & 0xFFFF
        checksum_recv = struct.unpack('>H', packet[20:22])[0]
        if checksum_calc != checksum_recv:
            self.get_logger().warn(
                f'checksum不符(算出{checksum_calc}, 收到{checksum_recv})，丟棄這包')
            return

        fx, fy, fz, tx, ty, tz = struct.unpack('>6h', packet[8:20])

        raw = {
            'fx': fx / SENSITIVITY['fx'], 'fy': fy / SENSITIVITY['fy'],
            'fz': fz / SENSITIVITY['fz'],
            'tx': tx / SENSITIVITY['tx'], 'ty': ty / SENSITIVITY['ty'],
            'tz': tz / SENSITIVITY['tz'],
        }

        if not self._offset_ready:
            self._accumulate_offset(raw)
            return

        debiased = {axis: raw[axis] - self._offset[axis] for axis in self._axes}
        filtered = self._apply_moving_average(debiased)
        if filtered is None:
            return

        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.wrench.force.x = filtered['fx']
        msg.wrench.force.y = filtered['fy']
        msg.wrench.force.z = filtered['fz']
        msg.wrench.torque.x = filtered['tx']
        msg.wrench.torque.y = filtered['ty']
        msg.wrench.torque.z = filtered['tz']
        self.pub.publish(msg)

    def _accumulate_offset(self, raw: dict):
        """收集開機(或剛歸零)後的前offset_samples筆原始讀值，湊滿後
        取平均當成軟體offset。若這次收集是action觸發的(_zero_pending_ack)，
        完成時把樣本數丟進_zero_done_q通知等待中的execute_callback。"""
        for axis in self._axes:
            self._offset_buffer[axis].append(raw[axis])

        if len(self._offset_buffer['fx']) < self.offset_samples:
            return

        for axis in self._axes:
            samples = self._offset_buffer[axis]
            self._offset[axis] = sum(samples) / len(samples)
            samples.clear()
        self._offset_ready = True
        self.get_logger().info(
            f'offset收集完成({self.offset_samples}筆)：' +
            ', '.join(f'{axis}={self._offset[axis]:.4f}' for axis in self._axes))

        if self._zero_pending_ack:
            self._zero_pending_ack = False
            self._zero_done_q.put(self.offset_samples)

    def _apply_moving_average(self, raw: dict):
        out = {}
        for axis, value in raw.items():
            h = self._history[axis]
            h.append(value)
            out[axis] = sum(h) / len(h)

        any_history = next(iter(self._history.values()))
        if len(any_history) < self.filter_window:
            return None
        return out


def main(args=None):
    rclpy.init(args=args)
    node = OptoForceNode()
    # 至少2個thread：timer(_poll)跟action execute_callback要能同時跑，
    # execute_callback才不會在等_zero_done_q時卡住_poll。
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
