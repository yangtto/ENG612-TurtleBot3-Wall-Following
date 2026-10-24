import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class DualPIDWallFollower(Node):

    def __init__(self):
        super().__init__('dual_pid_wall_follower')

        # Desired distance from left wall
        self.target_distance = 0.30

        # PID 1: Angular velocity control
        self.kp_ang = 1.0
        self.ki_ang = 0.01
        self.kd_ang = 0.50

        # PID 2: Linear velocity control
        # Controls speed based on wall-distance error magnitude
        self.kp_lin = 0.30
        self.ki_lin = 0.02
        self.kd_lin = 0.02

        # Speed settings
        self.nominal_speed = 0.10
        self.min_speed = 0.03
        self.max_speed = 0.12
        self.max_angular_speed = 0.8

        # Corner detection
        self.front_stop_distance = 0.35

        # Angular PID variables
        self.ang_integral = 0.0
        self.ang_previous_error = 0.0

        # Linear PID variables
        self.lin_integral = 0.0
        self.lin_previous_error = 0.0

        self.previous_time = None

        # Navigation states
        self.state = 'FOLLOW_WALL'
        self.current_yaw = None
        self.turn_start_yaw = None

        # ROS 2 publisher
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        # ROS 2 subscribers
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.get_logger().info(
            'Dual PID wall follower started'
        )

    def odom_callback(self, msg):

        q = msg.pose.pose.orientation

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)

        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

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

    def reset_pid(self):
        self.ang_integral = 0.0
        self.ang_previous_error = 0.0
        self.lin_integral = 0.0
        self.lin_previous_error = 0.0
        self.previous_time = None

    def scan_callback(self, msg):

        left_distance = self.get_distance(msg, 90)
        front_distance = self.get_distance(msg, 0)

        cmd = Twist()

        if self.current_yaw is None:
            self.cmd_pub.publish(cmd)
            return

        # STATE 1: TURN RIGHT
        if self.state == 'TURN_RIGHT':

            angle_turned = (
   			    self.turn_start_yaw - self.current_yaw + math.pi
			   ) % (2 * math.pi) - math.pi
			   
            remaining = math.pi / 2 - angle_turned

            if remaining > 0.04:

                cmd.angular.z = -min(
                    0.5,
                    max(0.12, remaining * 1.5)
                )

                self.cmd_pub.publish(cmd)
                return

            self.cmd_pub.publish(Twist())

            self.state = 'FOLLOW_WALL'
            self.turn_start_yaw = None
            self.reset_pid()

            self.get_logger().info(
                '90-degree right turn completed'
            )
            return

        # Stop if LiDAR measurements are unavailable
        if left_distance is None or front_distance is None:
            self.cmd_pub.publish(cmd)
            return

        # STATE 2: CORNER DETECTION
        if front_distance < self.front_stop_distance:

            self.cmd_pub.publish(cmd)

            self.state = 'TURN_RIGHT'
            self.turn_start_yaw = self.current_yaw

            self.get_logger().info(
                'Corner detected - turning right'
            )
            return

        # Calculate sampling interval
        current_time = time.monotonic()

        if self.previous_time is None:
            dt = 0.2
        else:
            dt = current_time - self.previous_time

        self.previous_time = current_time
        dt = max(dt, 0.001)

        # -------------------------------------
        # PID 1: ANGULAR VELOCITY
        # -------------------------------------

        distance_error = (
            left_distance - self.target_distance
        )

        self.ang_integral += distance_error * dt

        self.ang_integral = max(
            -1.0, min(1.0, self.ang_integral)
        )

        ang_derivative = (
            distance_error - self.ang_previous_error
        ) / dt

        angular_velocity = (
            self.kp_ang * distance_error
            + self.ki_ang * self.ang_integral
            + self.kd_ang * ang_derivative
        )

        angular_velocity = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, angular_velocity)
        )

        # -------------------------------------
        # PID 2: LINEAR VELOCITY
        # -------------------------------------

        # Desired speed decreases as wall error increases
        speed_error = -abs(distance_error)

        self.lin_integral += speed_error * dt

        self.lin_integral = max(
            -1.0, min(0.0, self.lin_integral)
        )

        lin_derivative = (
            speed_error - self.lin_previous_error
        ) / dt

        speed_adjustment = (
            self.kp_lin * speed_error
            + self.ki_lin * self.lin_integral
            + self.kd_lin * lin_derivative
        )

        linear_velocity = (
            self.nominal_speed + speed_adjustment
        )

        linear_velocity = max(
            self.min_speed,
            min(self.max_speed, linear_velocity)
        )

        # Publish robot velocities
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity

        self.cmd_pub.publish(cmd)

        # Update previous errors
        self.ang_previous_error = distance_error
        self.lin_previous_error = speed_error

        self.get_logger().info(
            f'Left: {left_distance:.3f} m | '
            f'Error: {distance_error:.3f} m | '
            f'Linear: {linear_velocity:.3f} m/s | '
            f'Angular: {angular_velocity:.3f} rad/s'
        )


def main(args=None):

    rclpy.init(args=args)
    node = DualPIDWallFollower()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
