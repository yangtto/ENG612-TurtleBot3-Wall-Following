import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarTest(Node):

    def __init__(self):
        super().__init__('lidar_test')

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.get_logger().info('LiDAR test started')

    def scan_callback(self, msg):

        def get_distance(angle_deg):
            readings = []

            for i, distance in enumerate(msg.ranges):
                angle = math.degrees(
                    msg.angle_min + i * msg.angle_increment
                )

                difference = (angle - angle_deg + 180) % 360 - 180

                if abs(difference) <= 3:
                    if (
                        math.isfinite(distance)
                        and msg.range_min <= distance <= msg.range_max
                    ):
                        readings.append(distance)

            if readings:
                return sum(readings) / len(readings)

            return None

        front = get_distance(0)
        left = get_distance(90)
        right = get_distance(-90)

        def display(value):
            return f'{value:.3f}' if value is not None else 'N/A'

        self.get_logger().info(
            f'Front: {display(front)} m | '
            f'Left: {display(left)} m | '
            f'Right: {display(right)} m'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
