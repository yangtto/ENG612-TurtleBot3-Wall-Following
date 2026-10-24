#!/usr/bin/env python3
"""ENG612 Project 2 -- physical-robot adaptation of your earlier group code.

LEFT-wall following at 0.30 m, using the ROS 2 /scan and /odom topics.
Angular PID controls yaw rate; linear PID controls odometry-measured speed.

Based on your Pasted text(20261023-092508).txt, retaining its class name,
PID gains, parameter names, odometry speed feedback, and CSV columns.
ROS 2 Humble / geometry_msgs/Twist. Intended for left-wall following inside
the rectangular laboratory arena, including clockwise inside corners.

Run: python3 dual_pid_group.py
Single PID: python3 dual_pid_group.py --ros-args -p controller:=single
Monitor: python3 dual_pid_group.py --ros-args -p enable_motion:=false

Starting gains are NOT hardware-validated. Start aligned with a wall on the
LEFT at approximately 0.30 m from the LiDAR. Stop teleop/other controllers.
Ctrl+C requests zero velocity. On a latched fault, check the cause and restart.
"""

import csv
import math
import signal
import statistics
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan


def clamp(value, low, high):
    """Limit a numeric value to an inclusive interval."""
    return max(low, min(high, value))


def angle_difference(target, current):
    """Signed shortest yaw difference, wrapped to [-pi, pi]."""
    return math.atan2(math.sin(target - current), math.cos(target - current))


class ScanReaderLeft(Node):
    def __init__(self):
        super().__init__('scan_reader_left')

        # Navigation thresholds: metres, except turn speeds (rad/s).
        self.target = 0.30
        self.front_limit = 0.35
        self.front_rearm = 0.45
        self.front_left_limit = 0.22  # Minimum diagonal clearance, both sides.
        self.max_turn_speed = 0.80
        self.wall_slowdown_error = 0.15
        self.front_slowdown_start = 0.65
        self.emergency_distance = 0.18
        self.corner_crawl_speed = 0.025
        self.maximum_wall_distance = 0.65

        # Original starting values; NOT validated PID tuning results.
        self.defaults = {
            'kp': 1.0,
            'ki': 0.01,
            'kd': 0.50,
            'kp_linear': 0.30,
            'ki_linear': 0.04,
            'kd_linear': 0.02,
            'minimum_speed': 0.07,
            'nominal_speed': 0.09,
            'maximum_speed': 0.10,
            'controller': 'dual',
            'enable_motion': True,
        }
        for name, default in self.defaults.items():
            self.declare_parameter(name, default)
        settings = self.read_settings()
        problem = self.validate_settings(settings)
        if problem:
            raise ValueError(problem)
        for name, value in settings.items():
            setattr(self, name, value)

        self.yaw = None
        self.measured_speed = None
        self.last_scan = None
        self.last_odom = None
        self.state = 'WAITING'
        self.state_since = time.monotonic()
        self.turn_target = None
        self.corner_armed = True
        self.started = False
        self.ready_scans = 0
        self.invalid_scans = 0
        self.last_notice = 0.0
        self.last_status = 0.0
        self.last_command = Twist()
        self.turn_count = 0
        self.tilt = None
        self.reset_pid()

        # Keep the group's original CSV measurement and tuning logs.
        folder = Path.home() / 'wall_follower_logs'
        folder.mkdir(parents=True, exist_ok=True)
        run_name = datetime.now().strftime(f'left_{self.controller}_pid_%Y%m%d_%H%M%S_%f')
        self.csv_path = folder / f'{run_name}.csv'
        self.csv_file = self.csv_path.open('w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'scan_time_s', 'front_m', 'left_m', 'desired_m', 'error_m',
            'linear_m_s', 'angular_rad_s', 'state', 'front_left_m', 'yaw_deg',
            'speed_reference_m_s', 'measured_speed_m_s', 'speed_error_m_s',
            'kp', 'ki', 'kd', 'kp_linear', 'ki_linear', 'kd_linear',
            'minimum_speed', 'nominal_speed', 'maximum_speed',
            'controller', 'enable_motion',
        ])
        (folder / f'{run_name}.settings.txt').write_text(
            '\n'.join(f'{name}={getattr(self, name)}' for name in self.defaults) + '\n',
            encoding='utf-8',
        )
        self.change_log = (folder / f'{run_name}.changes.csv').open(
            'w', newline='', encoding='utf-8'
        )
        self.change_writer = csv.writer(self.change_log)
        self.change_writer.writerow(['wall_time_s', 'parameter', 'new_value'])
        self.change_log.flush()

        self.pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )
        self.add_on_set_parameters_callback(self.validate_parameters)
        self.create_timer(0.1, self.watchdog)
        self.get_logger().info('LEFT wall follower: ' + self.controller + ' PID mode')
        self.get_logger().info(
            f'Target={self.target:.2f} m; nominal={self.nominal_speed:.3f} m/s'
        )
        self.get_logger().info(f'Logging measurements to: {self.csv_path}')
        self.get_logger().info(
            'Waiting for /scan and /odom; start with the LEFT wall about 0.30 m away.'
        )

    def read_settings(self):
        return {name: self.get_parameter(name).value for name in self.defaults}

    @staticmethod
    def validate_settings(settings):
        gains = ('kp', 'ki', 'kd', 'kp_linear', 'ki_linear', 'kd_linear')
        speeds = ('minimum_speed', 'nominal_speed', 'maximum_speed')
        for name in gains + speeds:
            value = settings[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                return f'{name} must be a finite number'
            if name in gains and not 0.0 <= value <= 10.0:
                return f'{name} must be between 0 and 10'
            if name in speeds and not 0.0 <= value <= 0.12:
                return f'{name} must be between 0 and 0.12 m/s'
        if not (settings['minimum_speed'] <= settings['nominal_speed']
                <= settings['maximum_speed']):
            return 'Require minimum_speed <= nominal_speed <= maximum_speed'
        if settings['controller'] not in ('single', 'dual'):
            return 'controller must be single or dual'
        if not isinstance(settings['enable_motion'], bool):
            return 'enable_motion must be a boolean'
        return ''

    def validate_parameters(self, params):
        """Validate before commit; do not mutate the node inside this callback."""
        candidate = self.read_settings()
        for parameter in params:
            if parameter.name in candidate:
                candidate[parameter.name] = parameter.value
        problem = self.validate_settings(candidate)
        return SetParametersResult(successful=not problem, reason=problem)

    def sync_parameters(self):
        """Apply successfully committed parameter changes on the next timer tick."""
        settings = self.read_settings()
        changes = {
            name: value
            for name, value in settings.items()
            if getattr(self, name) != value
        }
        if not changes:
            return
        for name, value in changes.items():
            setattr(self, name, value)
            self.change_writer.writerow([time.time(), name, value])
        self.change_log.flush()
        self.reset_pid()
        if 'enable_motion' in changes:
            # Changing the motion switch never bypasses a latched fault.
            self.send()
            if self.state != 'FAULT':
                self.started = False
                self.ready_scans = 0
                self.change_state('WAITING')
        self.get_logger().info(f'Parameters updated: {changes}')

    def send(self, linear=0.0, angular=0.0):
        """Publish motion command or a zero-velocity stop."""
        command = Twist()
        command.linear.x = float(clamp(linear, 0.0, self.maximum_speed))
        command.angular.z = float(clamp(
            angular, -self.max_turn_speed, self.max_turn_speed
        ))
        if not self.enable_motion:
            command = Twist()
        self.last_command = command
        self.pub.publish(command)
        return command

    def reset_pid(self):
        self.error_sum = 0.0
        self.last_error = None
        self.last_stamp = None
        self.filtered_d = 0.0
        self.speed_error_sum = 0.0
        self.last_speed_error = None
        self.last_speed_stamp = None
        self.filtered_speed_d = 0.0

    def change_state(self, new_state):
        if new_state != self.state:
            self.state = new_state
            self.state_since = time.monotonic()
            self.reset_pid()
            self.get_logger().info(f'State: {new_state}')

    def fault(self, message):
        """Latch a FAULT state; restart the node after checking the problem."""
        if self.state != 'FAULT':
            self.get_logger().error(message)
        self.change_state('FAULT')
        return self.send()

    def odom_callback(self, msg):
        if (self.started and self.last_odom is not None
                and time.monotonic() - self.last_odom > 1.0):
            self.fault('Gap in /odom exceeded 1.0 s; check bringup/network and restart.')
        q = msg.pose.pose.orientation
        norm = math.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)
        if not math.isfinite(norm) or norm < 1e-6:
            return
        qw, qx, qy, qz = (q.w/norm, q.x/norm, q.y/norm, q.z/norm)
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        roll = math.atan2(2.0 * (qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
        pitch = math.asin(clamp(2.0 * (qw*qy - qz*qx), -1.0, 1.0))
        speed = msg.twist.twist.linear.x
        if not (math.isfinite(yaw) and math.isfinite(speed)):
            return
        self.yaw = yaw
        self.tilt = max(abs(roll), abs(pitch))
        self.measured_speed = speed
        self.last_odom = time.monotonic()

    def sensors_ready(self, now):
        return (
            self.last_scan is not None
            and self.last_odom is not None
            and self.yaw is not None
            and self.measured_speed is not None
            and now - self.last_scan <= 1.0
            and now - self.last_odom <= 1.0
        )

    def watchdog(self):
        self.sync_parameters()
        now = time.monotonic()
        if self.state == 'FAULT':
            self.send()
            return
        if not self.sensors_ready(now):
            scan_age = 'never received' if self.last_scan is None else f'{now-self.last_scan:.2f} s old'
            odom_age = 'never received' if self.last_odom is None else f'{now-self.last_odom:.2f} s old'
            message = f'/scan: {scan_age}; /odom: {odom_age}'
            if self.started:
                self.fault('Sensor timeout: ' + message + '. Check bringup/network and restart.')
            else:
                self.notice('Waiting for sensors: ' + message)
                self.send()
                self.ready_scans = 0
            self.reset_pid()
        elif not self.enable_motion:
            self.send()
        elif self.tilt is not None and self.tilt > math.radians(15):
            self.fault('Excessive tilt; put the robot on the laboratory floor.')
        elif self.started:
            # Keep commands alive between ~5 Hz LiDAR updates. The watchdog
            # overrides this immediately if either sensor becomes stale.
            self.pub.publish(self.last_command)

    def notice(self, message):
        now = time.monotonic()
        if now - self.last_notice >= 2.0:
            self.last_notice = now
            self.get_logger().info(message)

    @staticmethod
    def distance(scan, angle_deg, width=3.0, nearest=False):
        """Read a sector at angle relative to robot front; left is +90 deg.

        Filter invalid ranges. Returns None if there are no valid beams in sector.
        """
        readings = []
        for i, value in enumerate(scan.ranges):
            if not math.isfinite(value):
                continue
            if not scan.range_min <= value <= scan.range_max:
                continue
            angle = math.degrees(scan.angle_min + i * scan.angle_increment)
            difference = (angle - angle_deg + 180.0) % 360.0 - 180.0
            if abs(difference) <= width:
                readings.append(value)
        if not readings:
            return None
        # Zeros, NaN, +/-inf and out-of-range returns are unknown, not free space.
        return min(readings) if nearest else statistics.median(readings)

    def angular_pid(self, error, stamp):
        if self.last_stamp is not None and self.last_error is not None:
            dt = stamp - self.last_stamp
            if 0.001 <= dt <= 1.0:
                self.error_sum = clamp(
                    self.error_sum + 0.5 * (error + self.last_error) * dt,
                    -0.20, 0.20,
                )
                derivative = (error - self.last_error) / dt
                alpha = dt / (0.30 + dt)  # First-order D low-pass filter.
                self.filtered_d += alpha * (derivative - self.filtered_d)
            else:
                self.error_sum = 0.0
                self.filtered_d = 0.0
        self.last_error = error
        self.last_stamp = stamp
        command = (
            self.kp * error
            + self.ki * self.error_sum
            + self.kd * self.filtered_d
        )
        return clamp(command, -self.max_turn_speed, self.max_turn_speed)

    def reference_speed(self, wall_error, front):
        """Slow for large wall errors and for a nearby front wall."""
        severity = clamp(
            abs(wall_error) / self.wall_slowdown_error, 0.0, 1.0
        )
        target_speed = max(
            self.minimum_speed,
            self.nominal_speed * (1.0 - 0.60 * severity),
        )
        if front < self.front_slowdown_start:
            front_fraction = clamp(
                (front - self.front_limit)
                / (self.front_slowdown_start - self.front_limit),
                0.0, 1.0,
            )
            # Preserve whichever slowdown is more restrictive.
            # A nonzero crawl avoids asymptotically stopping just before the
            # corner threshold (the previous nominal_speed*fraction did this).
            crawl = min(self.corner_crawl_speed, self.nominal_speed)
            front_cap = crawl + (self.nominal_speed - crawl) * front_fraction
            target_speed = min(target_speed, front_cap)
        return clamp(target_speed, 0.0, self.nominal_speed)

    def linear_pid(self, speed_reference, stamp):
        error = speed_reference - self.measured_speed
        if self.last_speed_stamp is not None and self.last_speed_error is not None:
            dt = stamp - self.last_speed_stamp
            if 0.001 <= dt <= 1.0:
                previous_integral = self.speed_error_sum
                self.speed_error_sum = clamp(
                    self.speed_error_sum
                    + 0.5 * (error + self.last_speed_error) * dt,
                    -0.10, 0.10,
                )
                derivative = (error - self.last_speed_error) / dt
                alpha = dt / (0.25 + dt)
                self.filtered_speed_d += alpha * (
                    derivative - self.filtered_speed_d
                )
                raw = (
                    speed_reference
                    + self.kp_linear * error
                    + self.ki_linear * self.speed_error_sum
                    + self.kd_linear * self.filtered_speed_d
                )
                # Do not accumulate error further into output saturation.
                if ((raw > self.maximum_speed and error > 0.0)
                        or (raw < 0.0 and error < 0.0)):
                    self.speed_error_sum = previous_integral
            else:
                self.speed_error_sum = 0.0
                self.filtered_speed_d = 0.0
        self.last_speed_error = error
        self.last_speed_stamp = stamp
        raw = (
            speed_reference
            + self.kp_linear * error
            + self.ki_linear * self.speed_error_sum
            + self.kd_linear * self.filtered_speed_d
        )
        return clamp(raw, 0.0, self.maximum_speed)

    def record(self, stamp, front, left, front_left, cmd, speed_ref=0.0):
        self.writer.writerow([
            stamp, front, left, self.target,
            None if left is None else left - self.target,
            cmd.linear.x, cmd.angular.z, self.state, front_left,
            None if self.yaw is None else math.degrees(self.yaw),
            speed_ref, self.measured_speed,
            None if self.measured_speed is None
            else speed_ref - self.measured_speed,
            self.kp, self.ki, self.kd,
            self.kp_linear, self.ki_linear, self.kd_linear,
            self.minimum_speed, self.nominal_speed, self.maximum_speed,
            self.controller, self.enable_motion,
        ])
        self.csv_file.flush()

    def scan_callback(self, scan):
        now = time.monotonic()
        if self.last_scan is not None and self.started and now-self.last_scan > 1.0:
            self.fault('Gap in /scan exceeded 1.0 s; check LiDAR/network and restart.')
        self.last_scan = now
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        front = self.distance(scan, 0.0, width=12.0, nearest=True)
        left = self.distance(scan, 90.0, width=3.0)
        front_left = self.distance(scan, 45.0, width=12.0, nearest=True)
        front_right = self.distance(scan, -45.0, width=12.0, nearest=True)
        right = self.distance(scan, -90.0, width=10.0, nearest=True)
        valid_ranges = [r for r in scan.ranges if math.isfinite(r)
                        and max(0.0, scan.range_min) < r <= scan.range_max]
        nearest = min(valid_ranges) if valid_ranges else None

        if self.state == 'FAULT' or not self.sensors_ready(now):
            cmd = self.send()
            self.reset_pid()
            self.record(stamp, front, left, front_left, cmd)
            return

        if (front is None or front_left is None or front_right is None
                or right is None or left is None):
            self.invalid_scans += 1
            self.ready_scans = 0
            if self.started and self.invalid_scans >= 3:
                cmd = self.fault('Three scans lack valid ranges in a required sector.')
            else:
                self.notice('Missing valid LiDAR sector: front/left/right/diagonals. Stopped.')
                cmd = self.send()
            self.reset_pid()
            self.record(stamp, front, left, front_left, cmd)
            return
        self.invalid_scans = 0

        if self.tilt is None or self.tilt > math.radians(15):
            cmd = self.fault('Missing orientation or excessive tilt.')
            self.record(stamp, front, left, front_left, cmd)
            return
        if nearest is not None and nearest < self.emergency_distance:
            cmd = self.fault(f'Obstacle {nearest:.3f} m from LiDAR; clearance below 0.18 m.')
            self.record(stamp, front, left, front_left, cmd)
            return

        if not self.enable_motion:
            self.notice(f'MONITOR: left={left:.3f} m; front={front:.3f} m; right={right:.3f} m')
            self.record(stamp, front, left, front_left, self.send())
            return

        if self.state == 'WAITING':
            if not 0.22 <= left <= 0.45:
                self.ready_scans = 0
                self.notice(f'Place wall on LEFT at about 0.30 m. Current left={left:.3f} m.')
            else:
                self.ready_scans += 1
                if self.ready_scans >= 3:
                    self.started = True
                    self.change_state('FOLLOW')
            self.record(stamp, front, left, front_left, self.send())
            return

        if front >= self.front_rearm:
            self.corner_armed = True

        # Finish a clockwise 90-degree corner turn before resuming follow mode.
        if self.state == 'TURN_RIGHT':
            error = angle_difference(self.turn_target, self.yaw)
            if now - self.state_since > 20.0:
                cmd = self.fault('Right-turn timeout')
            elif front_right < self.front_left_limit:
                cmd = self.fault('Obstacle too close during right turn')
            elif abs(math.degrees(error)) <= 2.0:
                self.turn_target = None
                self.turn_count += 1
                self.get_logger().info(f'90-degree right turn {self.turn_count} completed')
                self.change_state('SETTLE')
                cmd = self.send()
            else:
                cmd = self.send(angular=clamp(1.5 * error, -0.40, 0.40))
            self.record(stamp, front, left, front_left, cmd)
            return

        if self.state == 'SETTLE':
            cmd = self.send()
            if now - self.state_since >= 0.4:
                if front < self.front_rearm or left > self.maximum_wall_distance:
                    cmd = self.fault('No clear next wall after turn; check placement/arena.')
                else:
                    self.change_state('FOLLOW')
            self.record(stamp, front, left, front_left, cmd)
            return

        if front <= self.front_limit:
            if (self.corner_armed and right >= 0.40
                    and front_right >= 0.28):
                self.corner_armed = False
                self.turn_target = self.yaw - math.pi / 2.0
                self.change_state('TURN_RIGHT')
                cmd = self.send()  # Stop forward travel before rotating.
            else:
                cmd = self.fault('Front obstacle: unable to start safe right turn')
            self.record(stamp, front, left, front_left, cmd)
            return

        # Extra diagonal obstacle clearance check during forward travel.
        if (front_left < self.front_left_limit
                or front_right < self.front_left_limit):
            cmd = self.fault('Diagonal obstacle too close')
            self.record(stamp, front, left, front_left, cmd)
            return

        # Correct moderate distance errors while moving gently. Spinning in
        # place changes the beam angle but cannot restore the lateral offset.
        if left > self.maximum_wall_distance:
            cmd = self.fault('Left wall lost (>0.65 m); reposition beside the wall.')
        elif self.state == 'FOLLOW':
            error = left - self.target  # Positive error: left wall is too far away.
            angular = clamp(self.angular_pid(error, now), -0.45, 0.45)
            speed_ref = self.reference_speed(error, front)
            if self.controller == 'dual':
                linear = self.linear_pid(speed_ref, now)
                # Speed PID cannot cancel the approach-to-corner slowdown.
                linear = min(linear, speed_ref + 0.015)
            else:
                linear = speed_ref
            cmd = self.send(linear=linear, angular=angular)
            self.record(stamp, front, left, front_left, cmd, speed_ref)
            if now - self.last_status >= 1.0:
                self.last_status = now
                self.get_logger().info(
                    f'FOLLOW | left={left:.3f} m | front={front:.3f} m | '
                    f'error={error:+.3f} m | v={cmd.linear.x:.3f} m/s | '
                    f'w={cmd.angular.z:+.3f} rad/s'
                )
            return
        else:
            cmd = self.send()

        self.record(stamp, front, left, front_left, cmd)

    def close_logs(self):
        self.csv_file.close()
        self.change_log.close()


def main(args=None):
    # Keep ROS active until our finally block has sent zero velocities.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    node = None
    try:
        node = ScanReaderLeft()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                if rclpy.ok():
                    for _ in range(4):
                        node.send()
                        time.sleep(0.05)
            finally:
                node.close_logs()
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
