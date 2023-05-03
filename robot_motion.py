import math
import time

class AnglePID:
    def __init__(self, Kp, Ki, Kd, setpoint_angle, min_output, max_output):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint_angle = setpoint_angle
        self.min_output = min_output
        self.max_output = max_output

        self.last_error = 0
        self.integral = 0
        self.last_time = time.monotonic()








def compute(self, measured_angle):
    # Calculate time since last iteration
    now = time.monotonic()
    dt = now - self.last_time

    # Calculate angle error
    angle_error = self.setpoint_angle - measured_angle

    # Calculate integral and derivative terms
    self.integral += angle_error * dt
    derivative = (angle_error - self.last_error) / dt

    # Calculate output signal
    pid_output = self.Kp * angle_error + self.Ki * self.integral + self.Kd * derivative
    # Map PID output to motor speeds
    left_motor_speed = 1500 + pid_output
    right_motor_speed = 1500 - pid_output

    # Limit motor speeds to prevent saturation
    if left_motor_speed > self.max_output:
        left_motor_speed = self.max_output
    elif left_motor_speed < self.min_output:
        left_motor_speed = self.min_output
    if right_motor_speed > self.max_output:
        right_motor_speed = self.max_output
    elif right_motor_speed < self.min_output:
        right_motor_speed = self.min_output

    # Save variables for next iteration
    self.last_error = angle_error
    self.last_time = now
    return left_motor_speed, right_motor_speed


# def calculate_motor_speed(target_angle, target_distance):
#     # Define proportional and derivative gains
#     k_p = 1.1
#     k_d = 0.4
#     last_error_angle = 0
#     center_value = 1500
#     max_speed = 80
#     # Define initial error values
#
#     if target_angle > math.pi:
#         target_angle -= 2*math.pi
#     elif target_angle < -math.pi:
#         target_angle += 2*math.pi
#
#     # Compute proportional and derivative terms
#     p = k_p * target_angle
#     d = 1
#     # print("P = " + str(p) + " ")
#
#     # Compute motor speeds for left and right sides of the robot
#     if target_angle > 15:
#         speed_left = center_value - p - d
#         speed_right = center_value + p + d
#     elif target_angle < -15:
#         speed_left = center_value + p + d
#         speed_right = center_value - p - d
#     else:
#         speed_left = center_value - p - d
#         speed_right = center_value - p - d
#
#     # Limit motor speeds to range of 1000-2000
#     if target_angle is not None:
#         speed_left = max(center_value - max_speed, min(center_value + max_speed, speed_left))
#         speed_right = max(center_value - max_speed, min(center_value + max_speed, speed_right))
#     else:
#         speed_left = 1500
#         speed_right = 1500
#
#
#
#     # Update last error angle
#     last_error_angle = target_angle
#
#     # Return motor speeds for left and right sides of the robot
#     return int(speed_left), int(speed_right)
