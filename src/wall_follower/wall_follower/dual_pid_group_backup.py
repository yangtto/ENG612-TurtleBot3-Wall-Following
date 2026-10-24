#!/usr/bin/env python3

"""
ENG612 PROJECT 2
TurtleBot3 Burger - DUAL PID WALL FOLLOWING

Angular PID:
    Controls wall distance and steering.

Linear PID:
    Controls forward velocity using odometry.

Corner navigation:
    Similar to the original Single PID controller.
    Detect front wall -> stop -> turn 90 degrees right
    -> resume wall following.

Features:
    - Target left wall distance: 0.30 m
    - Angular and linear PID controllers
    - LiDAR wall-heading correction
    - Automatic 90-degree clockwise turns
    - Continuous operation until Ctrl+C or safety fault
    - One CSV file per experiment
    - Sensor timeout and obstacle protection

Commands:
    Monitor:
        python3 dual_pid_group_backup.py

    Move without automatic turns:
        python3 dual_pid_group_backup.py --move

    Move with automatic turns:
        python3 dual_pid_group_backup.py --move --corners
"""

import argparse
import csv
import math
import statistics
import time

from datetime import datetime
from pathlib import Path

import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def wrap(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle)
    )


# ============================================================
# DUAL PID CONTROLLER
# ============================================================

class DualPIDFollower(Node):

    TARGET = 0.30

    FRONT_STOP = 0.35
    FRONT_SLOW = 0.65
    FRONT_REARM = 0.48

    NEAR_LIMIT = 0.18
    MAX_WALL_DISTANCE = 0.65

    MAX_YAW = 0.45
    SENSOR_TIMEOUT = 1.0

    def __init__(self, move=False, corners=False):

        super().__init__('dual_pid_wall_follower')

        self.move = move
        self.corners = corners

        # ----------------------------------------------------
        # ANGULAR PID GAINS
        # ----------------------------------------------------

        self.kp = 1.30
        self.ki = 0.01
        self.kd = 0.65

        self.k_heading = 0.60

        # ----------------------------------------------------
        # LINEAR PID GAINS
        # ----------------------------------------------------

        self.kp_v = 0.30
        self.ki_v = 0.04
        self.kd_v = 0.02

        # ----------------------------------------------------
        # LINEAR SPEED SETTINGS
        # ----------------------------------------------------

        self.nominal_speed = 0.07
        self.maximum_speed = 0.085

        # ----------------------------------------------------
        # SENSOR VARIABLES
        # ----------------------------------------------------

        self.scan = {}

        self.last_scan = None
        self.last_odom = None

        self.yaw = None
        self.speed = None
        self.tilt = None

        # ----------------------------------------------------
        # NAVIGATION STATES
        # ----------------------------------------------------

        self.state = (
            'MONITOR' if not move else 'WAITING'
        )

        self.turn_start = None
        self.turn_start_time = None

        self.turn_count = 0

        self.corner_armed = True
        self.ready_count = 0

        self.fault_reason = ''
        self.last_report = 0.0

        self.last_cmd = Twist()

        self.reset_pid()

        # ----------------------------------------------------
        # CSV DATA LOGGING
        # ----------------------------------------------------

        folder = Path.home() / 'wall_follower_logs'

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = (
            'dual_pid_wall_' +
            datetime.now().strftime('%Y%m%d_%H%M%S') +
            '.csv'
        )

        self.csv_path = folder / filename

        self.csv_file = self.csv_path.open(
            'w',
            newline=''
        )

        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            'time_s',
            'state',
            'front_m',
            'left_m',
            'right_m',
            'front_left_m',
            'front_right_m',
            'wall_heading_deg',
            'desired_m',
            'distance_error_m',
            'speed_reference_m_s',
            'cmd_linear_m_s',
            'odom_linear_m_s',
            'cmd_angular_rad_s',
            'yaw_deg',
            'turns',
            'angular_kp',
            'angular_ki',
            'angular_kd',
            'heading_gain',
            'linear_kp',
            'linear_ki',
            'linear_kd',
            'reason'
        ])

        # ----------------------------------------------------
        # ROS 2 COMMUNICATION
        # ----------------------------------------------------

        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self.on_scan,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.on_odom,
            qos_profile_sensor_data
        )

        # Controller update at 10 Hz
        self.create_timer(0.1, self.step)

        self.get_logger().info(
            f'Mode={self.state}; CSV={self.csv_path}'
        )

    # ========================================================
    # LIDAR PROCESSING
    # ========================================================

    @staticmethod
    def sector(msg, degrees, width, nearest=False):

        values = []

        for i, value in enumerate(msg.ranges):

            if not math.isfinite(value):
                continue

            if not msg.range_min <= value <= msg.range_max:
                continue

            angle = math.degrees(
                msg.angle_min +
                i * msg.angle_increment
            )

            difference = (
                angle - degrees + 180
            ) % 360 - 180

            if abs(difference) <= width:
                values.append(value)

        if not values:
            return None

        if nearest:
            return min(values)

        return statistics.median(values)

    def on_scan(self, msg):

        self.scan = {

            'front': self.sector(
                msg, 0, 12, True
            ),

            'left': self.sector(
                msg, 90, 4
            ),

            'right': self.sector(
                msg, -90, 9, True
            ),

            'front_left': self.sector(
                msg, 45, 9, True
            ),

            'front_right': self.sector(
                msg, -45, 9, True
            ),

            'left_75': self.sector(
                msg, 75, 3
            ),

            'left_105': self.sector(
                msg, 105, 3
            )
        }

        self.last_scan = time.monotonic()

    # ========================================================
    # ODOMETRY PROCESSING
    # ========================================================

    def on_odom(self, msg):

        q = msg.pose.pose.orientation

        yaw = math.atan2(
            2 * (q.w*q.z + q.x*q.y),
            1 - 2 * (q.y*q.y + q.z*q.z)
        )

        speed = msg.twist.twist.linear.x

        roll = math.atan2(
            2 * (q.w*q.x + q.y*q.z),
            1 - 2 * (q.x*q.x + q.y*q.y)
        )

        pitch = math.asin(
            clamp(
                2 * (q.w*q.y - q.z*q.x),
                -1,
                1
            )
        )

        if all(map(
            math.isfinite,
            (yaw, speed, roll, pitch)
        )):

            self.yaw = yaw
            self.speed = speed

            self.tilt = max(
                abs(roll),
                abs(pitch)
            )

            self.last_odom = time.monotonic()

    # ========================================================
    # PID RESET
    # ========================================================

    def reset_pid(self):

        self.e_int = 0.0
        self.v_int = 0.0

        self.prev_e = None
        self.prev_v_error = None

        self.d_filt = 0.0
        self.v_d_filt = 0.0

        self.last_pid_time = None

    # ========================================================
    # STATE MANAGEMENT
    # ========================================================

    def set_state(self, state):

        if self.state != state:

            self.state = state

            self.reset_pid()

            self.get_logger().info(
                'State: ' + state
            )

    # ========================================================
    # VELOCITY COMMAND
    # ========================================================

    def send(self, linear=0.0, angular=0.0):

        cmd = Twist()

        cmd.linear.x = clamp(
            float(linear),
            0.0,
            self.maximum_speed
        )

        cmd.angular.z = clamp(
            float(angular),
            -self.MAX_YAW,
            self.MAX_YAW
        )

        if not self.move or self.state == 'FAULT':
            cmd = Twist()

        self.publisher.publish(cmd)

        self.last_cmd = cmd

        return cmd

    # ========================================================
    # SAFETY STOP
    # ========================================================

    def fault(self, reason):

        if self.state != 'FAULT':

            self.fault_reason = reason

            self.get_logger().error(
                'SAFETY STOP: ' + reason
            )

            self.set_state('FAULT')

        return self.send()

    # ========================================================
    # WALL HEADING ESTIMATION
    # ========================================================

    def heading(self):

        a = self.scan.get('left_75')
        b = self.scan.get('left_105')

        if a is None or b is None:
            return None

        dx = (
            a * math.cos(math.radians(75))
            - b * math.cos(math.radians(105))
        )

        dy = (
            a * math.sin(math.radians(75))
            - b * math.sin(math.radians(105))
        )

        if dx < 0.10:
            return None

        angle = math.atan2(dy, dx)

        if abs(angle) > math.radians(25):
            return None

        return angle

    # ========================================================
    # PID 1: ANGULAR CONTROL
    # ========================================================

    def angular_pid(self, error, heading, dt):

        self.e_int = clamp(
            self.e_int + error * dt,
            -0.15,
            0.15
        )

        if self.prev_e is None:
            derivative = 0.0
        else:
            derivative = (
                error - self.prev_e
            ) / dt

        alpha = dt / (0.25 + dt)

        self.d_filt += alpha * (
            derivative - self.d_filt
        )

        self.prev_e = error

        output = (
            self.kp * error
            + self.ki * self.e_int
            + self.kd * self.d_filt
            + self.k_heading * heading
        )

        return clamp(
            output,
            -self.MAX_YAW,
            self.MAX_YAW
        )

    # ========================================================
    # PID 2: LINEAR VELOCITY CONTROL
    # ========================================================

    def linear_pid(self, reference, dt):

        error = reference - self.speed

        self.v_int = clamp(
            self.v_int + error * dt,
            -0.10,
            0.10
        )

        if self.prev_v_error is None:
            derivative = 0.0
        else:
            derivative = (
                error - self.prev_v_error
            ) / dt

        alpha = dt / (0.25 + dt)

        self.v_d_filt += alpha * (
            derivative - self.v_d_filt
        )

        self.prev_v_error = error

        output = (
            reference
            + self.kp_v * error
            + self.ki_v * self.v_int
            + self.kd_v * self.v_d_filt
        )

        return clamp(
            output,
            0.0,
            self.maximum_speed
        )

    # ========================================================
    # CSV RECORDING
    # ========================================================

    def record(self, cmd, speed_ref=0.0):

        s = self.scan

        left = s.get('left')
        heading = self.heading()

        self.writer.writerow([

            time.time(),

            self.state,

            s.get('front'),
            left,
            s.get('right'),

            s.get('front_left'),
            s.get('front_right'),

            None if heading is None
            else math.degrees(heading),

            self.TARGET,

            None if left is None
            else left - self.TARGET,

            speed_ref,

            cmd.linear.x,
            self.speed,
            cmd.angular.z,

            None if self.yaw is None
            else math.degrees(self.yaw),

            self.turn_count,

            self.kp,
            self.ki,
            self.kd,
            self.k_heading,

            self.kp_v,
            self.ki_v,
            self.kd_v,

            self.fault_reason
        ])

        self.csv_file.flush()

    # ========================================================
    # PROTECTED CONTROLLER
    # ========================================================

    def step(self):

        try:
            self.control()

        except Exception as exc:

            self.record(
                self.fault(
                    'Controller exception: ' + str(exc)
                )
            )

    # ========================================================
    # MAIN CONTROL LOGIC
    # ========================================================

    def control(self):

        now = time.monotonic()

        # ----------------------------------------------------
        # FAULT STATE
        # ----------------------------------------------------

        if self.state == 'FAULT':

            self.record(self.send())
            return

        # ----------------------------------------------------
        # MONITOR MODE
        # ----------------------------------------------------

        if self.state == 'MONITOR':

            if now - self.last_report >= 1.0:

                self.last_report = now

                self.get_logger().info(
                    f'MONITOR | {self.scan}'
                )

            self.record(self.send())
            return

        # ----------------------------------------------------
        # SENSOR TIMEOUT PROTECTION
        # ----------------------------------------------------

        if (
            self.last_scan is None
            or self.last_odom is None
            or now - self.last_scan > self.SENSOR_TIMEOUT
            or now - self.last_odom > self.SENSOR_TIMEOUT
        ):

            if self.state == 'WAITING':
                self.record(self.send())

            else:
                self.record(
                    self.fault(
                        'LiDAR or odometry timeout'
                    )
                )

            return

        # ----------------------------------------------------
        # TILT PROTECTION
        # ----------------------------------------------------

        if (
            self.tilt is None
            or self.tilt > math.radians(15)
        ):

            self.record(
                self.fault('Excessive tilt')
            )

            return

        s = self.scan

        required = (
            'front',
            'left',
            'right',
            'front_left',
            'front_right'
        )

        if any(
            s.get(key) is None
            for key in required
        ):

            self.record(
                self.fault('Missing required LiDAR sector')
            )

            return

        front = s['front']
        left = s['left']

        # ----------------------------------------------------
        # OBSTACLE CLEARANCE
        # ----------------------------------------------------

        if min(
            front,
            left,
            s['front_left'],
            s['front_right']
        ) < self.NEAR_LIMIT:

            self.record(
                self.fault(
                    'Obstacle clearance below 0.18 m'
                )
            )

            return

        # ----------------------------------------------------
        # WAIT FOR VALID INITIAL POSITION
        # ----------------------------------------------------

        if self.state == 'WAITING':

            if (
                0.23 <= left <= 0.40
                and front > self.FRONT_STOP
            ):

                self.ready_count += 1

                if self.ready_count >= 3:
                    self.set_state('FOLLOW')

            else:
                self.ready_count = 0

            self.record(self.send())
            return

        # ====================================================
        # CORNER NAVIGATION - SAME GENERAL APPROACH AS
        # THE ORIGINAL SINGLE PID CONTROLLER
        # ====================================================

        if self.state == 'TURN_RIGHT':

            if now - self.turn_start_time > 15.0:

                self.record(
                    self.fault('Right-turn timeout')
                )

                return

            if s['front_right'] < 0.22:

                self.record(
                    self.fault(
                        'Insufficient clearance while turning'
                    )
                )

                return

            # Calculate clockwise angle turned
            turned = -wrap(
                self.yaw - self.turn_start
            )

            remaining = math.pi / 2 - turned

            # Turn completed
            if abs(remaining) <= math.radians(3):

                self.turn_count += 1

                self.get_logger().info(
                    f'Right turn {self.turn_count} completed'
                )

                # Resume Dual PID wall following
                self.set_state('FOLLOW')

                cmd = self.send()

            else:

                # Negative angular velocity means clockwise
                angular = -clamp(
                    1.5 * remaining,
                    -0.35,
                    0.35
                )

                cmd = self.send(
                    0.0,
                    angular
                )

            self.record(cmd)
            return

        # ----------------------------------------------------
        # CORNER REARMING
        # ----------------------------------------------------

        if front >= self.FRONT_REARM:
            self.corner_armed = True

        # ----------------------------------------------------
        # AUTOMATIC CORNER DETECTION
        # ----------------------------------------------------

        if front <= self.FRONT_STOP:

            if not self.corners:

                self.record(
                    self.fault(
                        'Front wall reached; corners disabled'
                    )
                )

                return

            if not self.corner_armed:

                self.record(
                    self.fault(
                        'Front wall still close after turn'
                    )
                )

                return

            # Verify right-turn clearance
            if (
                s['right'] < 0.38
                or s['front_right'] < 0.25
            ):

                self.record(
                    self.fault(
                        'Right-turn clearance insufficient'
                    )
                )

                return

            self.corner_armed = False

            self.turn_start = self.yaw
            self.turn_start_time = now

            self.set_state('TURN_RIGHT')

            self.record(self.send())
            return

        # ----------------------------------------------------
        # LEFT WALL VALIDATION
        # ----------------------------------------------------

        if left > self.MAX_WALL_DISTANCE:

            self.record(
                self.fault(
                    'Left wall lost (>0.65 m)'
                )
            )

            return

        # ----------------------------------------------------
        # DIAGONAL OBSTACLE PROTECTION
        # ----------------------------------------------------

        if (
            s['front_left'] < 0.22
            or s['front_right'] < 0.22
        ):

            self.record(
                self.fault(
                    'Diagonal obstacle too close'
                )
            )

            return

        # ----------------------------------------------------
        # WALL HEADING
        # ----------------------------------------------------

        heading = self.heading()

        if heading is None:

            self.record(
                self.fault(
                    'Cannot estimate wall heading'
                )
            )

            return

        # ----------------------------------------------------
        # CONTROLLER SAMPLING TIME
        # ----------------------------------------------------

        if self.last_pid_time is None:

            dt = 0.1

        else:

            dt = clamp(
                now - self.last_pid_time,
                0.02,
                0.25
            )

        self.last_pid_time = now

        # ====================================================
        # PID 1: ANGULAR VELOCITY
        # ====================================================

        error = left - self.TARGET

        angular = self.angular_pid(
            error,
            heading,
            dt
        )

        # ====================================================
        # PID 2: LINEAR VELOCITY
        # ====================================================

        # Slow down for large wall-distance errors
        distance_factor = (
            1.0
            - 0.6 * clamp(
                abs(error) / 0.15,
                0.0,
                1.0
            )
        )

        # Slow down for poor wall alignment
        heading_factor = (
            1.0
            - 0.6 * clamp(
                abs(heading) / math.radians(20),
                0.0,
                1.0
            )
        )

        # Corner slowdown:
        # Keep the front factor between 0.5 and 1.0
        front_factor = clamp(
            (front - self.FRONT_STOP) /
            (self.FRONT_SLOW - self.FRONT_STOP),
            0.5,
            1.0
        )

        speed_ref = (
            self.nominal_speed
            * min(
                distance_factor,
                heading_factor,
                front_factor
            )
        )

        # Actual linear PID output
        linear = self.linear_pid(
            speed_ref,
            dt
        )

        # ----------------------------------------------------
        # PUBLISH BOTH PID OUTPUTS
        # ----------------------------------------------------

        cmd = self.send(
            linear,
            angular
        )

        # ----------------------------------------------------
        # RECORD DATA
        # ----------------------------------------------------

        self.record(
            cmd,
            speed_ref
        )

        # ----------------------------------------------------
        # DISPLAY CONTROLLER STATUS
        # ----------------------------------------------------

        if now - self.last_report >= 1.0:

            self.last_report = now

            self.get_logger().info(

                f'FOLLOW | '
                f'left={left:.3f} '
                f'front={front:.3f} '
                f'error={error:+.3f} '
                f'v_ref={speed_ref:.3f} '
                f'v_cmd={cmd.linear.x:.3f} '
                f'v_odom={self.speed:.3f} '
                f'w={cmd.angular.z:+.3f}'
            )

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown_safely(self):

        for _ in range(5):

            try:

                if rclpy.ok():

                    self.publisher.publish(Twist())

                    time.sleep(0.05)

            except Exception:
                break

        self.csv_file.close()


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--move',
        action='store_true'
    )

    parser.add_argument(
        '--corners',
        action='store_true'
    )

    args = parser.parse_args()

    if args.corners and not args.move:

        parser.error(
            '--corners requires --move'
        )

    rclpy.init()

    node = DualPIDFollower(
        move=args.move,
        corners=args.corners
    )

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.shutdown_safely()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
