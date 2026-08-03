from tm_msgs.srv import SetPositions,SetEvent,SendScript
from tm_msgs.msg import FeedbackState
from robotiq_85_msgs.msg import GripperCmd

import rclpy
from rclpy.node import Node
import time
import numpy as np
import queue
# 0.34 -0.47 0.19 target
# move 0.34 -0.47 0.3
# move 0.34 -0.47 0.19
# pick
# move 0.34 -0.47 0.3
# move 0.2 -0.3 0.3
# move 0.2 -0.3 0.19
# place

class ArmScript(Node):
    def __init__(self):
        super().__init__('arm_script')
        self.pos_cli = self.create_client(SetPositions, 'set_positions')
        self.event_cli = self.create_client(SetEvent, 'set_event')
        self.script_cli = self.create_client(SendScript, 'send_script')
        while not self.pos_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        while not self.event_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        while not self.script_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.gripper_pub = self.create_publisher(GripperCmd, '/gripper/cmd', 10)

        self.pos_sub = self.create_subscription(FeedbackState, 'feedback_states', self.pos_callback, 10)

        self.target_positions = np.array([0.2, -0.4, 0.35, 3.14159, 0.0, -1.57])
        self.current_positions = np.array([0.2, -0.4, 0.35, 3.14159, 0.0, -1.57])
        

    def pos_callback(self,msg):
        self.current_positions = msg.tool_pose
        # self.get_logger().info("Current Position: %s" % self.current_positions)

    def is_arrived(self,error=0.01):
        if sum((self.target_positions[i]-self.current_positions[i])**2 for i in range(3)) > error**2:
            return False
        return True

    def PVTEnter(self):
        self.send_script("PVTEnter(1)")

    def PVTPoint(self,postions=np.array([0.2, -0.4, 0.35, 3.14159, 0.0, -1.57]),
                 velocity=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
                 duration=1):
        self.target_positions = postions
        postions = postions*1000
        postions[3:] = postions[3:]*180/3.14159/1000.0
        postions = ",".join(str(x) for x in postions)
        velocity = velocity*1000
        velocity[3:] = velocity[3:]*180/3.14159/1000.0
        velocity = ",".join(str(x) for x in velocity)
        print("PVTPoint({}, {}, {})".format(postions, velocity, duration))
        self.send_script("PVTPoint({}, {}, {})".format(postions, velocity, duration))

    def PVTExit(self):
        self.send_script("PVTExit()")

    def send_request(self,positions=[0.2, -0.4, 0.35, 3.14159, 0.0, -1.57],
                     velocity=0.4, acc_time=0.1, blend_percentage=10, fine_goal=False):
        self.target_positions = positions

        set_positions_req = SetPositions.Request()
        set_positions_req.motion_type = SetPositions.Request.PTP_T
        set_positions_req.positions = positions
        set_positions_req.velocity = velocity
        set_positions_req.acc_time = acc_time
        set_positions_req.blend_percentage = blend_percentage
        set_positions_req.fine_goal = fine_goal
        future = self.pos_cli.call_async(set_positions_req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def send_gripper(self,gap=0.085):
        gap = gap if gap < 0.085 else 0.085
        gap = gap if gap > 0.0 else 0.0

        grip_msg = GripperCmd()
        grip_msg.emergency_release = False
        grip_msg.emergency_release_dir = 0
        grip_msg.stop = False
        grip_msg.position = gap
        grip_msg.speed = 0.1
        grip_msg.force = 1.0
        self.gripper_pub.publish(grip_msg)

    def send_event(self):
        set_event_req = SetEvent.Request()
        set_event_req.func = SetEvent.Request.STOP
        set_event_req.arg0 = 0
        set_event_req.arg1 = 0
        future = self.event_cli.call_async(set_event_req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def send_script(self,cmd):
        script_req =  SendScript.Request()
        script_req.id = "demo"
        script_req.script = cmd
        future = self.script_cli.call_async(script_req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main(args=None):
    rclpy.init(args=args)
    armScript = ArmScript()
    rclpy.spin_once(armScript)
    armScript.send_gripper(0.085)
    armScript.send_event()

    response = armScript.send_request([0.0, -0.3, 0.35, 3.14159, 0.0, -1.57])
    while not armScript.is_arrived():
        rclpy.spin_once(armScript)
    print("move",armScript.target_positions)

    HZ = 5
    armScript.PVTEnter()
    distance = 0.400 #(m)
    speed = 0.1 #(m/s)
    total_time = distance/speed

    duration = 1/(HZ/3)
    fragment_size = speed*duration
    p = 0
    print("total_time %.3f" % total_time,"fragment_size %.3f" % fragment_size ,"duration %.3f" % duration)
    last = time.time()
    while True:
        
        if p > distance:
            break
        rclpy.spin_once(armScript)
        if armScript.is_arrived(fragment_size):
            # speed += 0.02
            p += fragment_size
            armScript.PVTPoint(np.array([p, -0.3, 0.35, 3.14159, 0.0, -1.57]),np.array([speed, 0, 0, 0, 0, 0]),duration)
            print("move",end=" ")
            for pos in armScript.current_positions:
                print("%.3f"%pos,end=" ")
            print()

        while (time.time() - last) < (1/HZ):
            rclpy.spin_once(armScript)
            
        print(1/(time.time() - last))
        last = time.time()

        # rclpy.spin_once(armScript)
    # speed = 0.2
    # while True:
    #     positions = list(map(float, input("Positions: ").split()))
    #     if len(positions) == 3:
    #         positions = positions + [3.14159, 0.0, -1.57]
    #     if len(positions) == 1:
    #         positions[0] /= 100.0
    #         armScript.send_gripper(positions[0])
    #         continue
    #     if len(positions) == 0:
    #         armScript.send_event()
    #         continue
    #     duration = sum([(positions[i]-armScript.current_positions[i])**2 for i in range(3)])**0.5/speed
    #     response = armScript.PVTPoint(positions,[0, 0, 0, 0, 0, 0],1)
    #     armScript.get_logger().info("Response: %s" % response)


    armScript.PVTExit()

    rclpy.shutdown()

if __name__ == '__main__':
    main()