import csv
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class WallLogger(Node):

    def __init__(self, filename):

        super().__init__('wall_logger')

        self.filename = filename
        self.target_distance = 0.30

        self.cmd_linear = 0.0
        self.cmd_angular = 0.0

        self.odom_linear = None
        self.odom_angular = None

        self.x = None
        self.y = None
        self.yaw = None

        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            'sim_time_s',
            'wall_time_s',
            'left_distance_m',
            'front_distance_m',
            'error_m',
            'cmd_linear_m_s',
            'cmd_angular_rad_s',
            'odom_linear_m_s',
            'odom_angular_rad_s',
            'x_m',
            'y_m',
            'yaw_rad',
            'phase'
        ])

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.row_count = 0

        self.get_logger().info(
            f'Data logger started: {filename}'
        )

    def cmd_callback(self, msg):

        self.cmd_linear = msg.linear.x
        self.cmd_angular = msg.angular.z

    def odom_callback(self, msg):

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        self.odom_linear = msg.twist.twist.linear.x
        self.odom_angular = msg.twist.twist.angular.z

        q = msg.pose.pose.orientation

        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)

        self.yaw = math.atan2(siny, cosy)

    def get_distance(self, msg, angle_deg):

        readings = []

        for i, distance in enumerate(msg.ranges):

            angle = math.degrees(
                msg.angle_min + i * msg.angle_increment
            )

            difference = (
                angle - angle_deg + 180
            ) % 360 - 180

            if abs(difference) <= 3:

                if (
                    math.isfinite(distance)
                    and msg.range_min <= distance <= msg.range_max
                ):
                    readings.append(distance)

        if readings:
            return sum(readings) / len(readings)

        return None

    def scan_callback(self, msg):

        left = self.get_distance(msg, 90)
        front = self.get_distance(msg, 0)

        if left is None or front is None:
            return

        error = left - self.target_distance

        sim_time = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        if (
            abs(self.cmd_linear) < 0.005
            and abs(self.cmd_angular) > 0.05
        ):
            phase = 'TURNING'
        elif abs(self.cmd_linear) > 0.005:
            phase = 'FOLLOWING'
        else:
            phase = 'STOPPED'

        self.writer.writerow([
            sim_time,
            time.time(),
            left,
            front,
            error,
            self.cmd_linear,
            self.cmd_angular,
            self.odom_linear,
            self.odom_angular,
            self.x,
            self.y,
            self.yaw,
            phase
        ])

        self.row_count += 1

        if self.row_count % 25 == 0:

            self.file.flush()

            self.get_logger().info(
                f'Recorded {self.row_count} samples | '
                f'Left: {left:.3f} m | '
                f'Error: {error:.3f} m'
            )

    def close(self):

        if not self.file.closed:
            self.file.flush()
            self.file.close()


def main():

    if len(sys.argv) < 2:
        print('Usage: python3 wall_logger.py output.csv')
        return

    filename = sys.argv[1]

    rclpy.init()

    node = WallLogger(filename)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print(f'Data saved to: {filename}')


if __name__ == '__main__':
    main()
