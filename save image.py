import cv2 as cv
import numpy as np
import time
from utils import *
import robot_motion
import data_sender

# w=246cm h=176cm l=165cm

# Load the cascade classifier
mtx, dist, newcameramtx, roi = load_calib_data()
vid = camera_init()

enemy = ImageProcessor("enemy robot", True)
robot = ImageProcessor("Our robot")
time.sleep(2)

# Initialize variables for calculating fps
num_frames = 0
start_time = time.time()


try:
    data_sender.setupSerial(115200, "COM7")
except:
    print("cant connect to Serial port")

show_image = 0
while True:
    ret, frame = vid.read()
    # Get the current position of the video capture in milliseconds
    # pos_msec = vid.get(cv2.CAP_PROP_POS_MSEC)

    # frame = cv.rotate(frame, cv.ROTATE_180)
    # frame = show(frame, mtx, dist, newcameramtx, roi)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # find our robot
    robot.process_image(frame)
    enemy.process_image(frame)

    if not show_image:
        mask = robot.show_image()
        mask_enemy = enemy.show_image()
    else:
        mask = robot.create_mask()
        mask_enemy = enemy.create_mask()

    rect_center, _ = draw_min_rect(mask, frame, robot.window_name)
    enemy_rect_center, _ = draw_min_rect(mask_enemy, frame, enemy.window_name)

    # find_all_contours_mask(frame, mask)

    try:
        if enemy_rect_center is not (0, 0):
            # a, b = dot_product(contour_center, rect_center, enemy_rect_center)
            angle, distance = angle_between_and_direcrion(rect_center, enemy_rect_center, enemy_rect_center)

            cv.line(frame, enemy_rect_center, enemy_rect_center, (0, 255, 0), 2)

            txt = ("angle: {0} Distance: {1}").format(angle, distance)

            put_text_on_point(frame, angle, rect_center, 50)
    except:
        print("FUCK")
    # signal_left, signal_right = robot_motion.calculate_motor_speed(angle, distance)

    # Increment the number of frames
    num_frames += 1

    # Calculate the elapsed time since the start of the program
    elapsed_time = time.time() - start_time

    # Calculate the current fps
    current_fps = num_frames / elapsed_time

    # Display the current fps
    cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("target_image", frame)
    if cv.waitKey(1) == ord('q'):
        break

    if cv.waitKey(1) == ord('c') and show_image == 0:
        show_image = 1
        cv2.destroyWindow(robot.window_name)
        cv2.destroyWindow(enemy.window_name)
vid.release()
cv.destroyAllWindows()
