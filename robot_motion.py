import math

def calculate_motor_speed(target_angle, target_distance):
    # Define proportional and derivative gains
    k_p = 1.1
    k_d = 0.4
    last_error_angle = 0
    center_value = 1500
    max_speed = 50
    # Define initial error values

    if target_angle > math.pi:
        target_angle -= 2*math.pi
    elif target_angle < -math.pi:
        target_angle += 2*math.pi

    # Compute proportional and derivative terms
    p = k_p * target_angle
    d = 1
    # print("P = " + str(p) + " ")

    # Compute motor speeds for left and right sides of the robot
    if target_angle > 15:
        speed_left = center_value - p - d
        speed_right = center_value + p + d
    elif target_angle < -15:
        speed_left = center_value + p + d
        speed_right = center_value - p - d
    else:
        speed_left = center_value - p - d
        speed_right = center_value - p - d

    # Limit motor speeds to range of 1000-2000
    if target_angle is not None:
        speed_left = max(center_value - max_speed, min(center_value + max_speed, speed_left))
        speed_right = max(center_value - max_speed, min(center_value + max_speed, speed_right))
    else:
        speed_left = 1500
        speed_right = 1500



    # Update last error angle
    last_error_angle = target_angle

    # Return motor speeds for left and right sides of the robot
    return int(speed_left), int(speed_right)