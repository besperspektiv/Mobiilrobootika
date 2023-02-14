import atexit
import socket
import time

left_motor_speed = 0
right_motor_speed = 0

start = time.time()
HOST = "192.168.10.128"  # The server's hostname or IP address
PORT = 8888  # The port used by the server

def to_byte(info_to_byte):
    to_byte = bytes(str(info_to_byte), 'utf-8')
    print(info_to_byte)
    return to_byte

def write_force_to_file(force_1, force_2):
    with open('signal_left.txt', 'w') as f:
        f.write(str(force_1))

    with open('signal_right.txt', 'w') as f:
        f.write(str(force_2))

def exit_handler():
    write_force_to_file(0, 0)
    print('Force 0, 0')

atexit.register(exit_handler)

write_force_to_file(left_motor_speed, right_motor_speed)

def _map(x, in_min, in_max, out_min, out_max):
    return round((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min,2)

def robot_motion(angle = 0, distance = 0):

    center_value = 0
    max_speed = 2

    if angle > 2:
        mapped_left_motor_speed  = _map(abs(angle), -180, 180, center_value, max_speed)
        write_force_to_file(mapped_left_motor_speed, 0)

        try:
            """Send mapped_left_motor_speed via TCP"""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                s.sendall(to_byte((0 ,mapped_left_motor_speed)))
        except Exception as e:
            print(e)

        return print(mapped_left_motor_speed, 0, angle)
    if angle < -2:
        mapped_right_motor_speed = _map(abs(angle), -180, 180, center_value, max_speed)
        write_force_to_file(0, mapped_right_motor_speed)

        try:
            """Send mapped_right_motor_speed via TCP"""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                s.sendall(to_byte((mapped_right_motor_speed, 0)))
        except Exception as e:
            print(e)

        return print(0, mapped_right_motor_speed, angle)
    else:
        write_force_to_file(0, 0)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                s.sendall(to_byte((0,0)))
        except Exception as e:
            print(e)

        return print(0, 0, angle)

    # write_force_to_file(maped_left_motor_speed, maped_right_motor_speed)
