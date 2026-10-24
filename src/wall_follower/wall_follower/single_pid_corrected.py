#!/usr/bin/env python3
"""TurtleBot3 left-wall following: one angular PID + safe corner state machine.

Angular PID gains and forward speed are unchanged from the latest experiment.
Check the robot's actual geometry and test under direct supervision.
"""
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def angle_error(target, actual):
    return math.atan2(math.sin(target - actual), math.cos(target - actual))


class SinglePIDWallFollower(Node):
    def __init__(self):
        super().__init__('single_pid_wall_follower')

        # SAME gains, speeds and target as your most recent experiment.
        self.target_distance = 0.30
        self.kp, self.ki, self.kd = 1.40, 0.01, 0.65
        self.linear_speed = 0.07
        self.max_angular_speed = 0.80
        self.front_stop_distance = 0.35

        # Navigation/safety settings (not PID gains).
        self.front_clear_distance = 0.45
        self.max_wall_reacquire = 0.65
        self.min_wall_reacquire = 0.20
        self.emergency_distance = 0.16  # LiDAR-origin distance, not chassis clearance
        self.sensor_timeout = 1.0
        self.turn_timeout = 15.0
        self.settle_duration = 0.40
        self.reacquire_timeout = 5.0
        self.reacquire_speed = 0.035
        self.turn_speed_max = 0.40
        self.turn_complete_tolerance = math.radians(2.5)

        self.state = 'WAITING'
        self.state_since = time.monotonic()
        self.started = False
        self.yaw = None
        self.measured_speed = None
        self.last_odom_time = None
        self.last_scan_time = None
        self.last_scan = None
        self.turn_target = None
        self.turn_complete_count = 0
        self.ready_count = 0
        self.corner_armed = True
        self.previous_error = None
        self.previous_time = None
        self.integral = 0.0
        self.filtered_derivative = 0.0

        self.start_time = time.monotonic()
        folder = Path.home() / 'wall_follower_logs'
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = folder / f'single_pid_corrected_{stamp}.csv'
        self.csv_file = self.csv_path.open('w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'time_s', 'left_distance_m', 'front_distance_m', 'setpoint_m',
            'error_m', 'linear_velocity_m_s', 'angular_velocity_rad_s',
            'measured_speed_m_s', 'yaw_deg', 'state', 'kp', 'ki', 'kd'
        ])
        self.csv_file.flush()

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_callback,
                                 qos_profile_sensor_data)
        self.create_timer(0.1, self.watchdog)
        self.get_logger().info(
            f'Single angular PID: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}; '
            f'forward={self.linear_speed:.3f} m/s; target={self.target_distance:.2f} m')
        self.get_logger().info(f'CSV: {self.csv_path}')
        self.get_logger().info('WAITING: start with left wall ~0.30 m away.')

    def reset_pid(self):
        self.previous_error = None
        self.previous_time = None
        self.integral = 0.0
        self.filtered_derivative = 0.0

    def set_state(self, state):
        if state != self.state:
            self.state = state
            self.state_since = time.monotonic()
            self.reset_pid()
            self.get_logger().info(f'State: {state}')

    def send(self, linear=0.0, angular=0.0):
        cmd = Twist()
        cmd.linear.x = float(clamp(linear, 0.0, self.linear_speed))
        cmd.angular.z = float(clamp(angular, -self.max_angular_speed,
                                    self.max_angular_speed))
        self.publisher.publish(cmd)
        return cmd

    def stop_fault(self, reason):
        if self.state != 'FAULT':
            self.get_logger().error(f'STOPPED: {reason}; inspect and restart.')
            self.set_state('FAULT')
        return self.send()

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if norm < 1e-6 or not math.isfinite(norm):
            return
        w, x, y, z = q.w/norm, q.x/norm, q.y/norm, q.z/norm
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        speed = msg.twist.twist.linear.x
        if math.isfinite(yaw) and math.isfinite(speed):
            self.yaw = yaw
            self.measured_speed = speed
            self.last_odom_time = time.monotonic()

    @staticmethod
    def distance(msg, angle_deg, width_deg=3, nearest=False):
        values = []
        for i, value in enumerate(msg.ranges):
            if not math.isfinite(value) or value <= max(0.0, msg.range_min):
                continue
            if value > msg.range_max:
                continue
            deg = math.degrees(msg.angle_min + i*msg.angle_increment)
            diff = (deg-angle_deg+180) % 360 - 180
            if abs(diff) <= width_deg:
                values.append(value)
        if not values:
            return None
        if nearest:
            return min(values)
        values.sort()
        return values[len(values)//2]

    def record(self, left, front, cmd):
        self.writer.writerow([
            time.monotonic()-self.start_time,
            left, front, self.target_distance,
            left-self.target_distance if left is not None else None,
            cmd.linear.x, cmd.angular.z, self.measured_speed,
            math.degrees(self.yaw) if self.yaw is not None else None,
            self.state, self.kp, self.ki, self.kd
        ])
        self.csv_file.flush()

    def watchdog(self):
        now = time.monotonic()
        if self.state == 'FAULT':
            self.send()
            return
        scan_ok = (self.last_scan_time is not None and
                   now-self.last_scan_time < self.sensor_timeout)
        odom_ok = (self.last_odom_time is not None and
                   now-self.last_odom_time < self.sensor_timeout)
        if not (scan_ok and odom_ok):
            if self.started:
                self.stop_fault('/scan or /odom stopped updating')
            else:
                self.send()

    def angular_pid(self, error, now):
        if self.previous_time is None:
            derivative = 0.0  # No artificial derivative kick on restart/after turn.
        else:
            dt = now-self.previous_time
            if not 0.001 <= dt <= 1.0:
                derivative = 0.0
                self.filtered_derivative = 0.0
            else:
                self.integral = clamp(self.integral + error*dt, -0.5, 0.5)
                derivative = (error-self.previous_error)/dt
                alpha = dt/(0.20+dt)  # Filter noisy LiDAR derivative.
                self.filtered_derivative += alpha*(derivative-self.filtered_derivative)
        self.previous_time = now
        self.previous_error = error
        angular = (self.kp*error + self.ki*self.integral +
                   self.kd*self.filtered_derivative)
        return clamp(angular, -self.max_angular_speed,
                     self.max_angular_speed)

    def scan_callback(self, msg):
        now = time.monotonic()
        self.last_scan_time = now
        left = self.distance(msg, 90)
        front = self.distance(msg, 0, width_deg=10, nearest=True)
        front_left = self.distance(msg, 45, width_deg=8, nearest=True)
        front_right = self.distance(msg, -45, width_deg=8, nearest=True)
        valid = [r for r in msg.ranges if math.isfinite(r) and
                 max(0.0, msg.range_min) < r <= msg.range_max]
        nearest = min(valid) if valid else None

        def finish(cmd):
            self.record(left, front, cmd)

        if self.state == 'FAULT':
            finish(self.send())
            return
        if (self.last_odom_time is None or self.yaw is None or
                now-self.last_odom_time >= self.sensor_timeout):
            finish(self.stop_fault('Odometry unavailable') if self.started else self.send())
            return
        if left is None or front is None or front_left is None or front_right is None:
            finish(self.stop_fault('Missing LiDAR sector') if self.started else self.send())
            return
        if nearest is not None and nearest < self.emergency_distance:
            finish(self.stop_fault('Obstacle inside emergency LiDAR distance'))
            return
        if self.state == 'WAITING':
            if 0.22 <= left <= 0.45 and front >= self.front_clear_distance:
                self.ready_count += 1
                if self.ready_count >= 3:
                    self.started = True
                    self.set_state('FOLLOW_WALL')
            else:
                self.ready_count = 0
            finish(self.send())
            return

        if self.state == 'TURN_RIGHT':
            if now-self.state_since > self.turn_timeout:
                finish(self.stop_fault('Turn exceeded time limit'))
                return
            # Signed shortest heading difference: avoids the 360-degree modulo bug.
            remaining = angle_error(self.turn_target, self.yaw)
            if abs(remaining) <= self.turn_complete_tolerance:
                self.turn_complete_count += 1
                if self.turn_complete_count >= 2:
                    self.set_state('SETTLE')
                finish(self.send())
                return
            self.turn_complete_count = 0
            w = clamp(1.5*remaining, -self.turn_speed_max, self.turn_speed_max)
            # Stop if a heading estimate would demand an unexpected LEFT turn.
            if w > 0:
                finish(self.stop_fault('Unexpected heading during right turn'))
                return
            finish(self.send(angular=min(-0.10, w)))
            return

        if self.state == 'SETTLE':
            if now-self.state_since >= self.settle_duration:
                self.set_state('REACQUIRE_WALL')
            finish(self.send())
            return

        if self.state == 'REACQUIRE_WALL':
            # Correct wall offset with PID once the NEXT left wall is observable.
            # Exact 0.30 m spacing is NOT required before PID restarts.
            if front <= self.front_clear_distance:
                finish(self.stop_fault('Front path blocked after turn'))
                return
            if not self.min_wall_reacquire <= left <= self.max_wall_reacquire:
                finish(self.stop_fault('Next left wall not in safe acquisition range'))
                return
            if now-self.state_since > self.reacquire_timeout:
                finish(self.stop_fault('Could not reacquire next wall'))
                return
            self.set_state('FOLLOW_WALL')
            error = left-self.target_distance
            w = self.angular_pid(error, now)
            finish(self.send(self.reacquire_speed, w))
            return

        # FOLLOW_WALL: only ONE PID (for angular velocity).
        if front >= self.front_clear_distance:
            self.corner_armed = True
        if front <= self.front_stop_distance:
            if not self.corner_armed:
                finish(self.stop_fault('Front obstacle while corner not rearmed'))
                return
            if front_right < 0.22:
                finish(self.stop_fault('Not enough front-right clearance for right turn'))
                return
            self.corner_armed = False
            self.turn_target = angle_error(self.yaw-math.pi/2, 0.0)
            self.turn_complete_count = 0
            self.set_state('TURN_RIGHT')
            finish(self.send())
            return
        if left > self.max_wall_reacquire:
            finish(self.stop_fault('Left wall lost'))
            return
        if front_left < 0.20 or front_right < 0.20:
            finish(self.stop_fault('Diagonal obstacle too close'))
            return
        error = left-self.target_distance
        w = self.angular_pid(error, now)
        finish(self.send(self.linear_speed, w))

    def close_logs(self):
        if not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
        self.get_logger().info(f'CSV saved: {self.csv_path}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SinglePIDWallFollower()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                try:
                    for _ in range(5):
                        node.send()
                        time.sleep(0.05)
                except Exception:
                    pass
            node.close_logs()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
