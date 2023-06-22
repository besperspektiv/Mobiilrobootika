import cv2 as cv
from utils import *
import data_sender

mouse_y = 0
mouse_x = 0


def mouse_callback(event, x, y, flags, param):
    global mouse_y, mouse_x
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        print("Mouse moved at ({}, {})".format(mouse_x, mouse_y))

robot = ImageProcessor("Our robot")

serial = 1
try:
    data_sender.setupSerial(115200)
except:
    serial = 0
    print("cant connect to Serial port")

# Load the cascade classifier
# vid, ret = camera_init()
vid = index_cameras(width=620, height=480)
vid = vid[0]
vid.set(cv2.CAP_PROP_SETTINGS, 0)

width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

mtx, dist, newcameramtx, roi = load_calib_data(width, height)

# vid = cv2.VideoCapture(1)

detector = MovingObjectDetector(learning_rate=0.1)

start = time.time()
show_image = 0

max_speed_rotation = 255
pid_rotation = PIDController(kp=2, ki=0.02, kd=0.58, min_output=-max_speed_rotation,
                             max_output=max_speed_rotation, max_integral=10)
max_speed_movement = 300
pid_distance = PIDController(kp=1, ki=0, kd=0, min_output=-max_speed_movement,
                             max_output=max_speed_movement, max_integral=0)

flag = False
prev_pos = (0, 0)
prev_mouse_pos = (0, 0)

enemy_rect_center = (width / 2, height / 2)
while True:
    start = time.time()
    ret, frame = vid.read()
    # frame = cv.rotate(frame, cv.ROTATE_180)
    # frame = show(frame, mtx, dist, newcameramtx, roi)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # find our robot
    robot.process_image(frame)

    if not show_image:
        mask = robot.show_image()
    else:
        mask = robot.create_mask()
    try:
        botom_point, top_point = find_triangle(mask, frame)
    except:
        print("pask")
        botom_point, top_point = (0, 0), (0, 0)

    if botom_point != (0, 0) and top_point != (0, 0) and botom_point != None:
        """Enemy detection"""
        centroid = detector.detect(frame, botom_point)
        if centroid is not None:
            x, y = centroid

            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            current_pos = (int(x), int(y))
            if current_pos != prev_pos:
                print("enemy_rect_center")
                enemy_rect_center = (int(x), int(y))
            prev_pos = current_pos
        else:
            if not flag:
                x, y = width / 2, height / 2
                flag = True

    current_mouse_pos = (mouse_x, mouse_y)
    if current_mouse_pos != prev_mouse_pos:
        enemy_rect_center = current_mouse_pos
    prev_mouse_pos = current_mouse_pos

    try:
        a, b = dot_product(botom_point, top_point, enemy_rect_center)
        angle, distance = angle_between_and_direcrion(a, b)
        cv.line(frame, top_point, enemy_rect_center, (0, 255, 0), 2)

        txt = ("angle: {0} Distance: {1}").format(angle, distance)
        put_text_on_point(frame, angle, top_point, 30)

        if not math.isnan(angle):
            center_pos = 1500
            signal = pid_rotation.calculate(0, angle)
            signal_speed = pid_distance.calculate(0, distance)
            signal_speed = abs(signal_speed)
            if angle < 0:
                signal = signal * -1
            if abs(angle) > 80:
                signal_left = center_pos - signal
                signal_right = center_pos + signal
                pid_rotation.Kd = 0.58
                pid_rotation.Kp = 2
            else:
                # calculate the controller value based on the angle difference and the distance to the destination point
                controller_value = map(abs(angle), 0, 80, 0, 0.5) + map(distance, 0, 100, 0, 0.5)
                print(controller_value)
                # calculate the turn signal based on the controller value
                signal_turn = controller_value * signal
                signal_left = center_pos - signal_turn + signal_speed
                signal_right = center_pos + signal_turn + signal_speed

            # print(signal_left, signal_right, txt, signal_speed)
            try:
                if serial:
                    if distance > 50:
                        data_sender.send_signal_to_motors(int(signal_right), int(signal_left))
                    else:
                        data_sender.send_signal_to_motors(1500, 1500)
            except:
                print("Cant send SERIAL data")
                pass
    except:
        print("ERROR")

    cv.imshow("target_image", frame)
    cv2.setMouseCallback('target_image', mouse_callback)

    if cv.waitKey(1) == ord('q'):
        break

    if cv.waitKey(1) == ord('c') and show_image == 0:
        show_image = 1
        cv2.destroyWindow(robot.window_name)
    end = time.time()
    # print(end - start)
vid.release()
cv.destroyAllWindows()
