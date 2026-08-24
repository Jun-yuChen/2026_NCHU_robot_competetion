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
"""
import struct
import time
from collections import deque

import rclpy
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
        self.declare_parameter('filter_window', 30)
        self.declare_parameter('offset_samples', 100)
        port = self.get_parameter('port').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        window = self.get_parameter('filter_window').get_parameter_value().integer_value
        if window < 1:
            self.get_logger().warn(f'filter_window={window}無效，改用1(不濾波)')
            window = 1
        self.filter_window = window
        self._axes = ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')
        # 每軸各自一個deque，maxlen=window，超出視窗的舊樣本會自動被擠掉。
        # 用moving average是為了平滑高頻雜訊，不是為了濾除感測器過載這種
        # 真實的力值變化，所以window不要設太大，不然力訊號會被拖慢。
        self._history = {axis: deque(maxlen=self.filter_window) for axis in self._axes}

        # 軟體offset：硬體歸零(_send_zero_command)之外再加一層，開機後
        # 收前N筆原始讀值取平均當offset，之後每筆讀值都先減掉這個offset
        # 再進moving average濾波。offset_samples<=0代表關閉這個機制，
        # offset直接視為0(等於不做軟體歸零，只靠硬體歸零)。
        offset_samples = self.get_parameter('offset_samples').get_parameter_value().integer_value
        self.offset_samples = max(offset_samples, 0)
        self._offset = {axis: 0.0 for axis in self._axes}
        self._offset_buffer = {axis: [] for axis in self._axes}
        self._offset_ready = (self.offset_samples == 0)

        self.get_logger().info(f'開啟序列埠 {port} (1000000 baud)...')
        # timeout=0 -> 非阻塞讀取，read()立刻回傳目前有的資料(可能是空的)，
        # 不會卡住ROS2的timer callback。
        self.ser = serial.Serial(port, baudrate=1_000_000, timeout=0)

        # ⚠️ 感測器的歸零設定斷電就會重置(協定文件明講)，這支driver每次
        # 開機都是回到原始未歸零基準——Fz的校正係數只有4.09 counts/N
        # (比Fx/Fy粗糙近10倍)，同樣的未歸零基準值換算到Fz上看起來會被
        # 放大近10倍，容易誤以為是解析錯誤。開機時自動送一次硬體歸零，
        # Speed/Filter維持出廠預設(100Hz/15Hz)不變。
        # 送這道指令的當下感測器不能受力，不然歸零基準會歪掉。
        time.sleep(0.2)  # 給序列埠一點時間穩定，避免第一個bytes遺失
        self._send_zero_command()

        self.pub = self.create_publisher(WrenchStamped, 'optoforce/wrench', 10)
        self._buffer = bytearray()

        # 讓外部可以隨時觸發重新歸零，不用重開整個node——序列埠同一時間
        # 只能被一個process開著，所以不能做成另一支獨立腳本直接連序列埠，
        # 只能透過已經握有序列埠的這個node自己提供service。
        # 用法: ros2 service call /optoforce/zero std_srvs/srv/Trigger {}
        self.zero_srv = self.create_service(Trigger, 'optoforce/zero', self._zero_callback)

        # 感測器預設100Hz持續傳送，用200Hz輪詢確保不漏包。
        self.timer = self.create_timer(1.0 / 200.0, self._poll)

    def _send_zero_command(self, speed: int = 10, filt: int = 4):
        """送9-byte CONFIGURATION封包觸發硬體歸零(ZERO=255)，Speed=10
        (100Hz)、Filter=4(15Hz cutoff)維持出廠預設值不變，只是把它們
        重新寫一次(協定要求Speed/Filter/Zero一起送)。
        Checksum = 170+0+50+3+Speed+Filter+Zero (取UINT16)。
        """
        # Send zero = 0 first
        '''
        zero = 0
        checksum = (170 + 0 + 50 + 3 + speed + filt + zero) & 0xFFFF
        packet = bytes([170, 0, 50, 3, speed, filt, zero]) + struct.pack('>H', checksum)
        self.ser.write(packet)
        '''

        time.sleep(0.5)

        # Send zero command
        zero = 255
        checksum = (170 + 0 + 50 + 3 + speed + filt + zero) & 0xFFFF
        packet = bytes([170, 0, 50, 3, speed, filt, zero]) + struct.pack('>H', checksum)
        self.ser.write(packet)
        self.get_logger().info('已送出歸零指令')

    def _zero_callback(self, request, response):
        self._send_zero_command()
        # 歸零基準改變了，濾波視窗跟軟體offset裡殘留的舊樣本都是用舊
        # 基準算出來的，混進新基準的樣本會讓歸零後的前幾筆輸出出現
        # 過渡性的錯誤值，所以歸零時把兩者都清空重新累積。
        for h in self._history.values():
            h.clear()
        for axis in self._axes:
            self._offset[axis] = 0.0
            self._offset_buffer[axis].clear()
        self._offset_ready = (self.offset_samples == 0)
        response.success = True
        response.message = '已送出歸零指令(感測器歸零瞬間不能受力)'
        return response

    def _poll(self):
        n = self.ser.in_waiting
        if n:
            self._buffer += self.ser.read(n)

        while True:
            idx = self._buffer.find(HEADER)
            if idx < 0:
                # 沒找到header，只留最後3 bytes(可能是header被從中截斷)，
                # 避免雜訊資料讓buffer無限長大。
                if len(self._buffer) > 3:
                    del self._buffer[:-3]
                return
            if idx > 0:
                del self._buffer[:idx]  # 丟掉header前面的雜訊/上一包的殘餘
            if len(self._buffer) < PACKET_LEN:
                return  # 這包還沒收完整，等下一次poll再湊

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

        # packet[4:6]=sample counter, packet[6:8]=status(過載/感測器錯誤，
        # 見協定文件STATUS章節)，目前沒用到，之後要判斷過載可以在這裡加。
        fx, fy, fz, tx, ty, tz = struct.unpack('>6h', packet[8:20])

        raw = {
            'fx': fx / SENSITIVITY['fx'], 'fy': fy / SENSITIVITY['fy'],
            'fz': fz / SENSITIVITY['fz'],
            'tx': tx / SENSITIVITY['tx'], 'ty': ty / SENSITIVITY['ty'],
            'tz': tz / SENSITIVITY['tz'],
        }

        if not self._offset_ready:
            self._accumulate_offset(raw)
            return  # 還在收集offset樣本，這幾包直接丟棄不publish

        debiased = {axis: raw[axis] - self._offset[axis] for axis in self._axes}
        filtered = self._apply_moving_average(debiased)
        if filtered is None:
            return  # 視窗還沒填滿(開機/剛歸零後)，先不publish

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
        取平均當成軟體offset，之後每筆讀值都會先減掉它。"""
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

    def _apply_moving_average(self, raw: dict):
        """對6軸各自套用簡單移動平均。所有軸共用同一個累積節奏(每包
        一起append)，所以只要檢查其中一軸的deque長度就知道視窗是否
        填滿。視窗還沒填滿時(開機/剛歸零後的前幾包)回傳None，
        呼叫端據此丟棄不publish，避免用不足window的樣本數算出的
        平均值當成正式輸出。"""
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()