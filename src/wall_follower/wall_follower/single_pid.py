#!/usr/bin/env python3

import math
import time
import csv
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data

class SinglePIDWallFollower(Node):

    def __init__(self):
        super().__init__('single_pid_wall_follower')

        self.target_distance = 0.30

        # PID GAINS - ORIGINAL VALUES
        self.kp = 1.30
        self.ki = 0.01
        self.kd = 0.65

        # ORIGINAL ROBOT SETTINGS
        self.linear_speed = 0.07
        self.max_angular_speed = 0.8
        self.front_stop_distance = 0.35

        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = None

        self.state = 'FOLLOW_WALL'
        self.current_yaw = None
        self.turn_start_yaw = None

        # ODOMETRY MEASUREMENT
        self.measured_speed = None

        # EXPERIMENT START TIME
        self.start_time = time.monotonic()

        # CREATE CSV LOG
        log_folder = Path.home() / 'wall_follower_logs'
        log_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.csv_path = (
            log_folder / f'single_pid_{timestamp}.csv'
        )

        self.csv_file = open(
            self.csv_path,
            'w',
            newline=''
        )

        self.csv_writer = csv.writer(self.csv_file)

        # CSV COLUMN HEADINGS
        self.csv_writer.writerow([
            'time_s',
            'left_distance_m',
            'front_distance_m',
            'setpoint_m',
            'error_m',
            'linear_velocity_m_s',
            'angular_velocity_rad_s',
            'measured_speed_m_s',
            'yaw_deg',
            'state',
            'kp',
            'ki',
            'kd'
        ])

        self.csv_file.flush()

        # ROS PUBLISHER
        self.publisher = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        # ROS SUBSCRIBERS
        self.create_subscription(
    	    LaserScan,
    	    '/scan',
       	    self.scan_callback,
    	    qos_profile_sensor_data
	)

        self.create_subscription(
    	    Odometry,
    	    '/odom',
    	    self.odom_callback,
    	    qos_profile_sensor_data
	)	

        self.get_logger().info(
            'Single PID wall follower started'
        )

        self.get_logger().info(
            f'CSV logging started: {self.csv_path}'
        )

        self.get_logger().info(
            f'PID gains: Kp={self.kp}, '
            f'Ki={self.ki}, Kd={self.kd}'
        )

    # -----------------------------------
    # ODOMETRY CALLBACK
    # -----------------------------------

    def odom_callback(self, msg):

        q = msg.pose.pose.orientation

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)

        cosy_cosp = 1 - 2 * (
            q.y * q.y + q.z * q.z
        )

        self.current_yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )

        self.measured_speed = (
            msg.twist.twist.linear.x
        )

    # -----------------------------------
    # LIDAR DISTANCE
    # -----------------------------------

    def get_distance(self, msg, angle_deg):

        readings = []

        for i, distance in enumerate(msg.ranges):

            angle = math.degrees(
                msg.angle_min +
                i * msg.angle_increment
            )

            difference = (
                angle - angle_deg + 180
            ) % 360 - 180

            if abs(difference) <= 3:

                if (
                    math.isfinite(distance)
                    and msg.range_min <=
                    distance <= msg.range_max
                ):
                    readings.append(distance)

        if readings:
            return sum(readings) / len(readings)

        return None

    # -----------------------------------
    # CSV LOGGING
    # -----------------------------------

    def log_data(
        self,
        left_distance,
        front_distance,
        error,
        cmd
    ):

        elapsed_time = (
            time.monotonic() - self.start_time
        )

        yaw_deg = (
            math.degrees(self.current_yaw)
            if self.current_yaw is not None
            else None
        )

        self.csv_writer.writerow([
            elapsed_time,
            left_distance,
            front_distance,
            self.target_distance,
            error,
            cmd.linear.x,
            cmd.angular.z,
            self.measured_speed,
            yaw_deg,
            self.state,
            self.kp,
            self.ki,
            self.kd
        ])

        self.csv_file.flush()

    # -----------------------------------
    # LASER SCAN CALLBACK
    # -----------------------------------

    def scan_callback(self, msg):

        left_distance = self.get_distance(
            msg, 90
        )

        front_distance = self.get_distance(
            msg, 0
        )

        cmd = Twist()

        error = (
            left_distance - self.target_distance
            if left_distance is not None
            else None
        )

        if self.current_yaw is None:

            self.publisher.publish(cmd)

            self.log_data(
                left_distance,
                front_distance,
                error,
                cmd
            )

            return

        # -----------------------------------
        # RIGHT TURN
        # -----------------------------------

        if self.state == 'TURN_RIGHT':

            if self.turn_start_yaw is None:
                self.turn_start_yaw = self.current_yaw

            angle_turned = (
                self.turn_start_yaw -
                self.current_yaw
            ) % (2 * math.pi)

            remaining = (
                math.pi / 2 - angle_turned
            )

            if remaining > 0.04:

                cmd.angular.z = -min(
                    0.5,
                    max(0.12, remaining * 1.5)
                )

                self.publisher.publish(cmd)

                self.log_data(
                    left_distance,
                    front_distance,
                    error,
                    cmd
                )

                return

            self.publisher.publish(Twist())

            self.state = 'FOLLOW_WALL'
            self.turn_start_yaw = None
            self.previous_time = None
            self.previous_error = 0.0
            self.integral = 0.0

            self.log_data(
                left_distance,
                front_distance,
                error,
                Twist()
            )

            self.get_logger().info(
                '90-degree right turn completed'
            )

            return

        # -----------------------------------
        # CHECK LIDAR
        # -----------------------------------

        if (
            left_distance is None
            or front_distance is None
        ):

            self.publisher.publish(cmd)

            self.log_data(
                left_distance,
                front_distance,
                error,
                cmd
            )

            return

        # -----------------------------------
        # CORNER DETECTION
        # -----------------------------------

        if (
            front_distance <
            self.front_stop_distance
        ):

            self.publisher.publish(cmd)

            self.state = 'TURN_RIGHT'

            self.turn_start_yaw = self.current_yaw

            self.log_data(
                left_distance,
                front_distance,
                error,
                cmd
            )

            self.get_logger().info(
                'Corner detected - turning right'
            )

            return

        # -----------------------------------
        # PID CONTROLLER
        # -----------------------------------

        current_time = time.monotonic()

        if self.previous_time is None:
            dt = 0.2
        else:
            dt = (
                current_time - self.previous_time
            )

        self.previous_time = current_time

        dt = max(dt, 0.001)

        error = (
            left_distance -
            self.target_distance
        )

        # INTEGRAL
        self.integral += error * dt

        self.integral = max(
            -1.0,
            min(1.0, self.integral)
        )

        # DERIVATIVE
        derivative = (
            error - self.previous_error
        ) / dt

        # PID OUTPUT
        angular_velocity = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        # ANGULAR VELOCITY LIMIT
        angular_velocity = max(
            -self.max_angular_speed,
            min(
                self.max_angular_speed,
                angular_velocity
            )
        )

        cmd.linear.x = self.linear_speed
        cmd.angular.z = angular_velocity

        # SEND COMMAND
        self.publisher.publish(cmd)

        # RECORD DATA
        self.log_data(
            left_distance,
            front_distance,
            error,
            cmd
        )

        self.previous_error = error

        # TERMINAL OUTPUT
        self.get_logger().info(
            f'Left: {left_distance:.3f} m | '
            f'Error: {error:+.3f} m | '
            f'Linear: {cmd.linear.x:.3f} m/s | '
            f'Angular: {angular_velocity:+.3f} rad/s'
        )

    # -----------------------------------
    # CLOSE CSV
    # -----------------------------------

    def close_logs(self):

        if not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()

        self.get_logger().info(
            f'Experiment saved: {self.csv_path}'
        )


# -----------------------------------
# MAIN
# -----------------------------------

def main(args=None):

    rclpy.init(args=args)

    node = SinglePIDWallFollower()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        # STOP ROBOT
        for _ in range(5):
            node.publisher.publish(Twist())
            time.sleep(0.05)

        node.close_logs()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
