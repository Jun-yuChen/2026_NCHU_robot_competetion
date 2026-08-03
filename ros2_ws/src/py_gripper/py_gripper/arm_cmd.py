
import rclpy.time_source
from tm_msgs.srv import SetPositions, SetEvent, SetIO
from tm_msgs.msg import FeedbackState

import rclpy
from rclpy.node import Node
import time
# 0.34 -0.47 0.19 target
# move 0.34 -0.47 0.3
# move 0.34 -0.47 0.19
# pick
# move 0.34 -0.47 0.3
# move 0.2 -0.3 0.3
# move 0.2 -0.3 0.19
# place

class ArmCmd(Node):
    def __init__(self):
        super().__init__('arm_cmd')
        self.pos_cli = self.create_client(SetPositions, 'set_positions')
        self.event_cli = self.create_client(SetEvent, 'set_event')
        self.io_cli = self.create_client(SetIO, '/set_io')
        self.pos_sub = self.create_subscription(FeedbackState, 'feedback_states', self.pos_callback, 10)

        while not self.pos_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        while not self.event_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.set_positions_req = SetPositions.Request()
        self.set_event_req = SetEvent.Request()
        self.target_positions = [0.2, -0.4, 0.35, 3.14159, 0.0, -1.57]
        self.current_positions = [0.2, -0.4, 0.35, 3.14159, 0.0, -1.57]
        
    def pos_callback(self,msg):
        self.current_positions = msg.tool_pose
        # self.get_logger().info("Current Position: %s" % self.current_positions)

    def is_arrived(self,error=0.01):
        if sum((self.target_positions[i]-self.current_positions[i])**2 for i in range(3)) > error**2:
            return False
        return True

    def set_position(self,positions=[0.2, -0.4, 0.35, 3.14159, 0.0, -1.57], velocity=0.1, acc_time=0.5, blend_percentage=100, fine_goal=False):
        
        self.target_positions = positions
        print(self.target_positions)
        set_positions_req = SetPositions.Request()
        set_positions_req.motion_type = SetPositions.Request.LINE_T
        set_positions_req.positions = positions
        set_positions_req.velocity = velocity
        set_positions_req.acc_time = acc_time
        set_positions_req.blend_percentage = blend_percentage
        set_positions_req.fine_goal = fine_goal
        future = self.pos_cli.call_async(set_positions_req)
        return future.result()

    def set_io(self,state=0):
        res = SetIO.Request()
        res.module = 1  # IO module 1
        res.type = 1    # digital IO
        res.pin = 0     # pin 0
        res.state = float(state)
        self.io_cli.call_async(res)

    def set_gripper(self, state=0):
        self.set_io(state=state)

    def send_event(self):
        rclpy.spin_once(self)
        self.set_event_req = SetEvent.Request()
        self.set_event_req.func = SetEvent.Request.STOP
        self.set_event_req.arg0 = 0
        self.set_event_req.arg1 = 0
        future = self.event_cli.call_async(self.set_event_req)
        return future.result()

    def wait_until_arrived(self, error=0.01, timeout=30.0, poll_hz=20.0):
        period = 1.0 / poll_hz
        start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=period)  # lets pos_callback update current_positions
            if self.is_arrived(error):
                return True
            if timeout is not None and (time.time() - start) > timeout:
                self.get_logger().warn("Timeout waiting for arm to arrive")
                return False  
        return False


def main(args=None):
    rclpy.init(args=args)
    armCmd = ArmCmd()

    gripper = 0
    armCmd.set_gripper(gripper)

    while True:
        positions = list(map(float, input("Positions: ").split()))
        if len(positions) == 3:
            positions = positions + [3.14159, 0.0, 3.14]

        if len(positions) == 0:
            armCmd.send_event()
            
        response = armCmd.set_position(positions)

        # status = armCmd.wait_until_arrived()
        # if status != True:
        #     print("timeout for reach target")

        armCmd.get_logger().info("Response: %s" % response)

        if gripper == 0:
            gripper = 1
        else:
            gripper = 0

        armCmd.set_gripper(gripper)
        print(f"Gripper state: {gripper}")



    rclpy.spin(armCmd)
    rclpy.shutdown()


if __name__ == '__main__':
    main()