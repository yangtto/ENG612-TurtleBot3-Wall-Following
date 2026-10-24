import math
import time
import sys
import statistics

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


# ============================================================
# PHYSICAL TURTLEBOT3 BURGER - DUAL PID WALL FOLLOWER
# ============================================================

# Safety settings
ENABLE_CORNER_TURNS = False

TARGET_DISTANCE = 0.30

# Angular PID gains
KP_ANG = 1.0
KI_ANG = 0.01
KD_ANG = 0.50

# Linear PID gains
KP_LIN = 0.30
KI_LIN = 0.02
KD_LIN = 0.02

# Reduced speeds for physical testing
NOMINAL_SPEED = 0.04
MIN_SPEED = 0.015
MAX_SPEED = 0.04

MAX_ANGULAR_SPEED = 0.30
TURN_SPEED = 0.25

# Safety distances
FRONT_STOP_DISTANCE = 0.50
MIN_VALID_DISTANCE = 0.12
MAX_VALID_DISTANCE = 3.50

# Maximum acceptable wall-following error
MAX_WALL_ERROR = 0.15

# Maximum sensor age
SENSOR_TIMEOUT = 0.70

# Maximum allowable robot tilt
MAX_TILT_DEGREES = 15.0


class DualPIDWallFollower(Node):

    def __init__(self, movement_enabled=False):

        super().__init__('dual_pid_wall_follower')

        self.movement_enabled = movement_enabled

        # Target
        self.target_distance = TARGET_DISTANCE

        # Angular PID
        self.kp_ang = KP_ANG
        self.ki_ang = KI_ANG
        self.kd_ang = KD_ANG

        # Linear PID
        self.kp_lin = KP_LIN
        self.ki_lin = KI_LIN
        self.kd_lin = KD_LIN

        # Controller variables
        self.ang_integral = 0.0
        self.ang_previous_error = None

        self.lin_integral = 0.0
        self.lin_previous_error = None

        self.previous_time = None

        # LiDAR variables
        self.left_distance = None
        self.front_distance = None
        self.last_scan_time = None

        # Odometry variables
        self.current_yaw = None
        self.last_odom_time = None
        self.tilt_degrees = 0.0

        # Navigation
        self.state = 'FOLLOW_WALL'

        self.turn_previous_yaw = None
        self.turn_progress = 0.0
        self.turn_count = 0
        self.turn_start_time = None
        self.pause_start_time = None

        # Status
        self.last_log_time = 0.0
        self.stop_reason = ''

        # ROS publisher
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Real LiDAR uses sensor-data QoS
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        # Odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data
        )

        # Controller frequency: 10 Hz
        self.timer = self.create_timer(
            0.1,
            self.control_step
        )

        self.get_logger().info(
            'Physical TurtleBot3 Dual PID Controller Started'
        )

        if movement_enabled:
            self.get_logger().warning(
                'MOVEMENT ENABLED - Keep emergency stop accessible'
            )
        else:
            self.get_logger().info(
                'MONITORING MODE - Movement disabled'
            )

    # ========================================================
    # LiDAR DISTANCE MEASUREMENT
    # ========================================================

    def get_distance(self, msg, angle_deg, width_deg=5):

        readings = []

        for i, distance in enumerate(msg.ranges):

            angle = math.degrees(
                msg.angle_min +
                i * msg.angle_increment
            )

            difference = (
                angle - angle_deg + 180
            ) % 360 - 180

            if abs(difference) <= width_deg:

                if (
                    math.isfinite(distance)
                    and MIN_VALID_DISTANCE < distance
                    and distance < MAX_VALID_DISTANCE
                ):
                    readings.append(distance)

        if not readings:
            return None

        # Front distance uses nearest valid obstacle
        if angle_deg == 0:
            return min(readings)

        # Median reduces measurement noise
        return statistics.median(readings)

    def scan_callback(self, msg):

        self.left_distance = self.get_distance(
            msg, 90, 5
        )

        self.front_distance = self.get_distance(
            msg, 0, 12
        )

        self.last_scan_time = time.monotonic()

    # ========================================================
    # ODOMETRY
    # ========================================================

    def odom_callback(self, msg):

        q = msg.pose.pose.orientation

        siny = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        yaw = math.atan2(siny, cosy)

        # Calculate roll
        sinr = 2.0 * (
            q.w * q.x +
            q.y * q.z
        )

        cosr = 1.0 - 2.0 * (
            q.x * q.x +
            q.y * q.y
        )

        roll = math.atan2(sinr, cosr)

        # Calculate pitch
        sinp = 2.0 * (
            q.w * q.y -
            q.z * q.x
        )

        pitch = math.asin(
            max(-1.0, min(1.0, sinp))
        )

        self.tilt_degrees = math.degrees(
            max(abs(roll), abs(pitch))
        )

        # Track clockwise corner rotation
        if self.state == 'TURN_RIGHT':

            if self.turn_previous_yaw is not None:

                delta = math.atan2(
                    math.sin(
                        yaw - self.turn_previous_yaw
                    ),
                    math.cos(
                        yaw - self.turn_previous_yaw
                    )
                )

                if abs(delta) < 0.5:

                    self.turn_progress -= delta

                    self.turn_progress = max(
                        0.0,
                        self.turn_progress
                    )

        self.turn_previous_yaw = yaw
        self.current_yaw = yaw

        self.last_odom_time = time.monotonic()

    # ========================================================
    # PID RESET
    # ========================================================

    def reset_pid(self):

        self.ang_integral = 0.0
        self.ang_previous_error = None

        self.lin_integral = 0.0
        self.lin_previous_error = None

        self.previous_time = None

    # ========================================================
    # STOP ROBOT
    # ========================================================

    def stop_robot(self):

        self.cmd_pub.publish(Twist())

    def safety_stop(self, reason):

        if self.state != 'SAFETY_STOP':

            self.get_logger().error(
                'SAFETY STOP: ' + reason
            )

        self.stop_reason = reason
        self.state = 'SAFETY_STOP'

        self.stop_robot()

    # ========================================================
    # MAIN CONTROLLER
    # ========================================================

    def control_step(self):

        now = time.monotonic()

        cmd = Twist()

        # Monitoring mode: never command motion
        if not self.movement_enabled:

            self.stop_robot()

            if now - self.last_log_time >= 1.0:

                self.last_log_time = now

                self.get_logger().info(
                    f'MONITORING | '
                    f'Left: {self.left_distance} | '
                    f'Front: {self.front_distance}'
                )

            return

        # -----------------------------------------------
        # SAFETY CHECKS
        # -----------------------------------------------

        if self.state == 'SAFETY_STOP':

            self.stop_robot()
            return

        if (
            self.last_scan_time is None
            or now - self.last_scan_time > SENSOR_TIMEOUT
        ):

            self.safety_stop(
                'LiDAR data unavailable or stale'
            )
            return

        if (
            self.last_odom_time is None
            or now - self.last_odom_time > SENSOR_TIMEOUT
        ):

            self.safety_stop(
                'Odometry unavailable or stale'
            )
            return

        if self.tilt_degrees > MAX_TILT_DEGREES:

            self.safety_stop(
                'Robot tilted excessively'
            )
            return

        if self.front_distance is None:

            self.safety_stop(
                'No valid front LiDAR measurement'
            )
            return

        if self.left_distance is None:

            self.safety_stop(
                'Left wall not detected'
            )
            return

        # -----------------------------------------------
        # TURN RIGHT
        # -----------------------------------------------

        if self.state == 'TURN_RIGHT':

            if (
                self.turn_start_time is None
                or now - self.turn_start_time > 10.0
            ):

                self.safety_stop(
                    'Corner turn timeout'
                )
                return

            remaining = (
                math.pi / 2 -
                self.turn_progress
            )

            if remaining > 0.04:

                cmd.linear.x = 0.0

                cmd.angular.z = -min(
                    TURN_SPEED,
                    max(
                        0.10,
                        remaining * 1.5
                    )
                )

                self.cmd_pub.publish(cmd)
                return

            self.stop_robot()

            self.turn_count += 1

            self.get_logger().info(
                f'Turn {self.turn_count} completed | '
                f'{math.degrees(self.turn_progress):.1f} degrees'
            )

            self.state = 'PAUSE_AFTER_TURN'

            self.pause_start_time = now

            return

        # -----------------------------------------------
        # PAUSE AFTER TURN
        # -----------------------------------------------

        if self.state == 'PAUSE_AFTER_TURN':

            self.stop_robot()

            if now - self.pause_start_time < 1.0:
                return

            if (
                self.front_distance <
                FRONT_STOP_DISTANCE
            ):

                self.safety_stop(
                    'Front obstruction remains after turn'
                )
                return

            self.reset_pid()

            self.state = 'FOLLOW_WALL'

            self.get_logger().info(
                'Resuming wall following'
            )

            return

        # -----------------------------------------------
        # FRONT OBSTACLE DETECTION
        # -----------------------------------------------

        if (
            self.front_distance <=
            FRONT_STOP_DISTANCE
        ):

            self.stop_robot()

            if ENABLE_CORNER_TURNS:

                self.state = 'TURN_RIGHT'

                self.turn_progress = 0.0
                self.turn_previous_yaw = self.current_yaw

                self.turn_start_time = now

                self.get_logger().info(
                    'Corner detected - turning right'
                )

            else:

                self.safety_stop(
                    'Front wall detected - corner turning disabled'
                )

            return

        # -----------------------------------------------
        # WALL-LOSS DETECTION
        # -----------------------------------------------

        distance_error = (
            self.left_distance -
            self.target_distance
        )

        if abs(distance_error) > MAX_WALL_ERROR:

            self.safety_stop(
                f'Wall tracking error too large: '
                f'{distance_error:.3f} m'
            )
            return

        # -----------------------------------------------
        # SAMPLING INTERVAL
        # -----------------------------------------------

        if self.previous_time is None:

            dt = 0.1

        else:

            dt = now - self.previous_time

        self.previous_time = now

        dt = max(0.001, min(dt, 0.5))

        # ===============================================
        # PID 1: ANGULAR VELOCITY
        # ===============================================

        self.ang_integral += (
            distance_error * dt
        )

        self.ang_integral = max(
            -0.5,
            min(0.5, self.ang_integral)
        )

        if self.ang_previous_error is None:

            ang_derivative = 0.0

        else:

            ang_derivative = (
                distance_error -
                self.ang_previous_error
            ) / dt

        angular_velocity = (
            self.kp_ang * distance_error
            + self.ki_ang * self.ang_integral
            + self.kd_ang * ang_derivative
        )

        angular_velocity = max(
            -MAX_ANGULAR_SPEED,
            min(
                MAX_ANGULAR_SPEED,
                angular_velocity
            )
        )

        # ===============================================
        # PID 2: LINEAR VELOCITY
        # ===============================================

        speed_error = -abs(distance_error)

        self.lin_integral += (
            speed_error * dt
        )

        self.lin_integral = max(
            -0.5,
            min(0.0, self.lin_integral)
        )

        if self.lin_previous_error is None:

            lin_derivative = 0.0

        else:

            lin_derivative = (
                speed_error -
                self.lin_previous_error
            ) / dt

        speed_adjustment = (
            self.kp_lin * speed_error
            + self.ki_lin * self.lin_integral
            + self.kd_lin * lin_derivative
        )

        linear_velocity = (
            NOMINAL_SPEED +
            speed_adjustment
        )

        linear_velocity = max(
            MIN_SPEED,
            min(
                MAX_SPEED,
                linear_velocity
            )
        )

        # -----------------------------------------------
        # PUBLISH VELOCITY COMMANDS
        # -----------------------------------------------

        cmd.linear.x = linear_velocity

        cmd.angular.z = angular_velocity

        self.cmd_pub.publish(cmd)

        # Save previous errors
        self.ang_previous_error = distance_error

        self.lin_previous_error = speed_error

        # Display once per second
        if now - self.last_log_time >= 1.0:

            self.last_log_time = now

            self.get_logger().info(
                f'Left: {self.left_distance:.3f} m | '
                f'Front: {self.front_distance:.3f} m | '
                f'Error: {distance_error:.3f} m | '
                f'Linear: {linear_velocity:.3f} m/s | '
                f'Angular: {angular_velocity:.3f} rad/s'
            )


# ============================================================
# MAIN
# ============================================================

def main():

    # Movement must be explicitly enabled
    movement_enabled = '--move' in sys.argv

    ros_args = [
        arg for arg in sys.argv
        if arg != '--move'
    ]

    rclpy.init(args=ros_args)

    node = DualPIDWallFollower(
        movement_enabled=movement_enabled
    )

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        # Attempt to stop robot before shutdown
        try:

            if rclpy.ok():

                for _ in range(5):
                    node.stop_robot()
                    time.sleep(0.1)

        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':

    main()
