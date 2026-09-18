import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class OptoForceZeroClient(Node):

    def __init__(self):
        super().__init__('optoforce_zero_client')

        # Create the service client
        self.client = self.create_client(
            Trigger,
            'optoforce/zero'
        )

    def zero_sensor(self):
        # Wait until the service is available
        self.get_logger().info('Waiting for optoforce/zero service...')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                'Service not available, waiting...'
            )

        self.get_logger().info('Calling optoforce/zero...')

        # Create and send the request
        request = Trigger.Request()
        future = self.client.call_async(request)

        # Wait for the service response
        rclpy.spin_until_future_complete(self, future)

        # Check whether the call itself succeeded
        if future.result() is None:
            self.get_logger().error(
                'Failed to call optoforce/zero service'
            )
            return False

        # Get the response
        response = future.result()

        # Check the service response
        if response.success and response.message.startswith('SENSOR_OK'):
            self.get_logger().info('Zero done')
            return True

        self.get_logger().error(
            f'Zero failed: {response.message}'
        )
        return False


def main(args=None):
    rclpy.init(args=args)

    node = OptoForceZeroClient()

    try:
        success = node.zero_sensor()

        if success:
            node.get_logger().info('Sensor zeroing completed successfully.')
        else:
            node.get_logger().error('Sensor zeroing failed.')

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()