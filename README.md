# ENG612 Project 2 – TurtleBot3 Wall Following

## Project Overview

This project implements autonomous left-wall following using
a TurtleBot3 Burger robot and ROS 2 Humble.

The target wall distance is 0.30 m.

Two control approaches were developed and evaluated:

1. Single PID – Angular velocity control with constant forward speed.
2. Dual PID – Angular and linear velocity control with adaptive speed regulation.

## Software and Hardware

- TurtleBot3 Burger
- ROS 2 Humble
- Ubuntu
- Python 3
- Gazebo Classic
- LiDAR and wheel odometry

## ROS 2 Topics

| Topic | Function |
|---|---|
| /scan | LiDAR measurements |
| /odom | Robot orientation and velocity |
| /cmd_vel | Robot movement commands |

## Final PID Settings

### Single PID

- Kp = 1.30
- Ki = 0.01
- Kd = 0.65
- Forward speed = 0.07 m/s

### Dual PID

Angular PID:
- Kp = 1.30
- Ki = 0.01
- Kd = 0.65

Linear PID:
- Kp = 0.30
- Ki = 0.04
- Kd = 0.02

Additional heading gain = 0.60

Nominal speed = 0.07 m/s

Maximum commanded speed = 0.085 m/s

## Running the Controllers

First, start TurtleBot3 bringup on the robot.

### Single PID

```bash
python3 src/wall_follower/wall_follower/single_pid.py
```

### Dual PID

```bash
python3 src/wall_follower/wall_follower/dual_pid_group_backup.py --move --corners
```

Position the robot approximately 0.30 m from the left wall.

Keep the robot supervised and ensure no other node is controlling /cmd_vel.

Press Ctrl+C to request a stop.

## Gazebo World

The custom simulation environment is stored at:

src/wall_follower/worlds/wall_following.world

## Experimental Results

| Metric | Single PID | Dual PID |
|---|---|---|
| RMSE | 1.02 cm | 1.15 cm |
| MAE | 0.77 cm | 1.09 cm |
| Within ±2 cm | 96.37% | 98.33% |

After excluding the initial positioning period, the Dual PID
achieved approximately 1.08 cm RMSE and 100% of wall-following
measurements within ±2 cm.

The final Dual PID experiment completed 20 clockwise turns.

## Contributors

- Group Member 1: [Name]
- Group Member 2: [Name]

## Acknowledgement

AI-assisted coding and debugging suggestions were reviewed,
modified and tested during project development.
