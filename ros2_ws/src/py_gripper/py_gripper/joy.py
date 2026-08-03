import time
import json
import queue
from pynput import keyboard
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np

W = set(['w'])
S = set(['s'])
A = set(['a'])
D = set(['d'])
WD = set(['w','d'])
WA = set(['w','a'])
SD = set(['s','d'])
SA = set(['s','a'])
UP = set(['p'])
DOWN = set([';'])
LEFT = set([keyboard.Key.left])
RIGHT = set([keyboard.Key.right])

class Joy(Node):
    def __init__(self):
        super().__init__('joy')
        self.currently_pressed = set()

        def on_release(key):
            try:
                self.currently_pressed.remove(key.char)
            except AttributeError:
                if key == keyboard.Key.space:
                    self.currently_pressed.remove(key)
                pass

        def on_press(key):
            try:
                self.currently_pressed.add(key.char)
            except AttributeError:
                if key == keyboard.Key.space:
                    self.currently_pressed.add(key)
                pass
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        self.grip = False
        self.joy_pub = self.create_publisher(Float64MultiArray,'/joy',30)
        self.create_timer(0.1, self.run)
            
        
    def run(self):
        msg = Float64MultiArray()
        pressBool = False
        data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        if 'w' in self.currently_pressed:
            data[1] = -1.0
            pressBool = True
            print('w')

        elif 's' in self.currently_pressed:
            data[1] = 1.0
            pressBool = True
            print('s')

        if 'a' in self.currently_pressed:
            data[0] = 1.0
            pressBool = True
            print('a')

        elif 'd' in self.currently_pressed:
            data[0] = -1.0
            pressBool = True
            print('d')

        if 'p' in self.currently_pressed:
            data[2] = 1.0
            pressBool = True
            print('up')
        elif ';' in self.currently_pressed:
            data[2] = -1.0
            pressBool = True
            print('down')
        
        if keyboard.Key.space in self.currently_pressed:
            if self.grip == True:
                data = [0.03]
                self.grip = False
                pressBool = True
            else:
                data = [0.085]
                self.grip = True
                pressBool = True
        # print(self.currently_pressed)
        if pressBool:
            msg.data = data
            print(msg.data)
            self.joy_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    joy = Joy()
    rclpy.spin(joy)
    joy.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()