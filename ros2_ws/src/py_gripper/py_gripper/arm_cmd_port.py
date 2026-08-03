
import rclpy.time_source
# Techman Robot 專用的訊息與服務
from tm_msgs.srv import SetPositions,SetEvent
from tm_msgs.msg import FeedbackState
# Robotiq 夾爪專用的訊息，用於控制夾爪開合。
from robotiq_85_msgs.msg import GripperCmd

# ROS 2 的 Python 介面核心
import rclpy
from rclpy.node import Node
import time

# 為了實現 Socket 通訊，需引入相關模組
import socket
import json
import threading # 引入多執行緒庫


class ArmCmd(Node):
    def __init__(self):
        # 建立 Service Client (用於發送指令)
        super().__init__('arm_cmd')
        self.pos_cli = self.create_client(SetPositions, 'set_positions')
        self.event_cli = self.create_client(SetEvent, 'set_event')

        # 建立 Publisher (用於控制夾爪)
        self.gripper_pub = self.create_publisher(GripperCmd, '/gripper/cmd', 10)

        # 建立 Subscriber (用於監聽手臂位置)
        self.pos_sub = self.create_subscription(FeedbackState, 'feedback_states', self.pos_callback, 10)

        # 等待 Service 連線成功
        while not self.pos_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        while not self.event_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        # 初始化變數
        self.set_positions_req = SetPositions.Request()
        self.set_event_req = SetEvent.Request()
        self.target_positions = [0.2, -0.4, 0.35, 3.14159, 0.0, -1.57]
        self.current_positions = [0.2, -0.4, 0.35, 3.14159, 0.0, -1.57]

        # --- 啟動 Socket Server ---
        self.start_socket_server()
    

    def start_socket_server(self):
        # 建立一個子執行緒來跑 Socket，避免卡住 ROS 的 Spin
        self.server_thread = threading.Thread(target=self._socket_worker)
        self.server_thread.daemon = True # 設定為守護執行緒，主程式結束時它也會自動結束
        self.server_thread.start()
        self.get_logger().info("Socket Server started on 127.0.0.1:65432")

    def _socket_worker(self):
        HOST = '0.0.0.0'  # 1. 修改：允許外部連線
        PORT = 65432
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            self.get_logger().info(f"Socket Server listening on {HOST}:{PORT}")
            
            while True:
                conn, addr = s.accept()
                with conn:
                    self.get_logger().info(f"Connected by {addr}")
                    
                    # 2. 修改：使用 file-like object 讀取，解決黏包問題
                    f = conn.makefile('r', encoding='utf-8')
                    
                    while True: # 3. 修改：保持連線 (長連線)
                        try:
                            # 讀取一行 (直到遇到 \n)
                            line = f.readline()
                            if not line: 
                                break # Client 斷線
                            
                            msg = json.loads(line.strip())
                            self.get_logger().info(f"Received: {msg}")

                            # --- 您的邏輯 (這部分保持原樣，寫得很好) ---
                            if "positions" in msg:
                                target = msg["positions"]
                                if len(target) == 6:
                                    self.send_request(target)
                                elif len(target) == 3:
                                    target = target + [3.14159, 0.0, -1.57]
                                    self.send_request(target)
                            
                            if "gripper" in msg:
                                val = float(msg["gripper"])
                                if val > 1.0: val /= 100.0
                                self.send_gripper(val)
                            # ----------------------------------------

                        except json.JSONDecodeError:
                            self.get_logger().error("JSON Error: Incomplete or invalid format")
                        except Exception as e:
                            self.get_logger().error(f"Socket error: {e}")
                            break
                    
                    self.get_logger().info(f"Disconnected from {addr}")
        

    # 當收到機器人回傳訊息時，更新當前座標
    def pos_callback(self,msg):
        self.current_positions = msg.tool_pose
        # self.get_logger().info("Current Position: %s" % self.current_positions)

    # 計算「目標位置」與「當前位置」的平方誤差和，判斷是否到達目標位置
    def is_arrived(self,error=0.01):
        if sum((self.target_positions[i]-self.current_positions[i])**2 for i in range(3)) > error**2:
            return False
        return True


    def send_request(self,positions=[0.2, -0.4, 0.35, 3.14159, 0.0, -1.57],
                     velocity=0.1, acc_time=0.5, blend_percentage=100, fine_goal=False):
        
        self.target_positions = positions
        print(self.target_positions)

        # 建構 Service Request
        set_positions_req = SetPositions.Request()
        set_positions_req.motion_type = SetPositions.Request.LINE_T # 直線運動
        set_positions_req.positions = positions
        set_positions_req.velocity = velocity   # 速度
        set_positions_req.acc_time = acc_time   # 加速時間
        set_positions_req.blend_percentage = blend_percentage   # 路徑平滑度
        set_positions_req.fine_goal = fine_goal

        # 非同步呼叫 Service
        future = self.pos_cli.call_async(set_positions_req)
        # rclpy.spin_until_future_complete(self, future)
        return future.result()
    

    def send_gripper(self,gap=0.085):
        # 限制夾爪開度在 0.0 (閉合) 到 0.085 (全開，8.5cm) 之間
        gap = gap if gap < 0.085 else 0.085
        gap = gap if gap > 0.0 else 0.0
        print(gap)

        # 建構夾爪訊息
        grip_msg = GripperCmd()
        grip_msg.emergency_release = False
        grip_msg.emergency_release_dir = 0
        grip_msg.stop = False
        grip_msg.position = gap
        grip_msg.speed = 0.1
        grip_msg.force = 1.0
        self.gripper_pub.publish(grip_msg)


    def send_event(self):
        rclpy.spin_once(self)
        self.set_event_req = SetEvent.Request()
        self.set_event_req.func = SetEvent.Request.STOP
        self.set_event_req.arg0 = 0
        self.set_event_req.arg1 = 0
        future = self.event_cli.call_async(self.set_event_req)
        # rclpy.spin_until_future_complete(self, future)
        return future.result()
    

def main(args=None):
    rclpy.init(args=args)
    armCmd = ArmCmd()
    
    # 初始化動作
    armCmd.get_logger().info("System Ready. Waiting for TCP commands...")
    armCmd.send_gripper(0.085)  # 初始化張開夾爪

    try:
        # 直接讓 ROS 進入 spin 狀態，這樣它才能處理 Service 回傳和 Socket 請求
        rclpy.spin(armCmd)
    except KeyboardInterrupt:
        pass
    finally:
        armCmd.destroy_node()
        rclpy.shutdown()

    rclpy.spin(armCmd)
    rclpy.shutdown()



if __name__ == '__main__':
    main()