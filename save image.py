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

serial = 1
try:
    data_sender.setupSerial(115200, "COM7")
except:
    serial = 0
    print("cant connect to Serial port")

show_image = 0
while True:
    ret, frame = vid.read()
    # frame = cv.rotate(frame, cv.ROTATE_180)
    frame = show(frame, mtx, dist, newcameramtx, roi)
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

    image_to_draw, rect_center, contour_center = draw_min_rect(mask, frame, robot.window_name)
    image_to_draw, enemy_rect_center, _ = draw_min_rect(mask_enemy, image_to_draw, enemy.window_name)

    a, b = dot_product(contour_center, rect_center, enemy_rect_center)
    angle, distance = angle_between_and_direcrion(a, b)

    cv.line(image_to_draw, rect_center, enemy_rect_center, (0, 255, 0), 2)

    txt = ("angle: {0} Distance: {1}").format(angle, distance)

    put_text_on_point(image_to_draw, angle, rect_center, 30)
    cv.imshow("target_image", image_to_draw)


    signal_left, signal_right = robot_motion.calculate_motor_speed(angle, distance)

    print(signal_left, signal_right, txt)

    if serial:
        if distance > 50:
            data_sender.send_signal_to_motors(signal_left, signal_right)
        else:
            data_sender.send_signal_to_motors(1500, 1500)

    if cv.waitKey(1) == ord('q'):
        break

    if cv.waitKey(1) == ord('c') and show_image == 0:
        show_image = 1
        cv2.destroyWindow(robot.window_name)
        cv2.destroyWindow(enemy.window_name)


vid.release()
cv.destroyAllWindows()
