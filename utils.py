from __future__ import print_function
from pygrabber.dshow_graph import FilterGraph
import cv2
import numpy as np
import math
import time

def read_owerwritted_image_from_path(image_path = "C:/Users/pikka/PycharmProjects/Mobiilrobootika/screenshot.png"):
    try:
        frame = cv2.imread(image_path)
        return frame
    except:
        frame = cv2.imread(image_path)
        return frame
        pass

def map(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low

def calculate_controller_value(angle, distance, max_distance):
    normalized_angle = angle / 180.0
    normalized_distance = distance / max_distance
    controller_value = normalized_angle * normalized_distance
    return controller_value * 1


target_camera = "c922 Pro Stream Webcam"
def index_cameras(width=1280, height=720, fps=60):
    graph = FilterGraph()
    usb_devices = graph.get_input_devices()
    print("USB devices {}".format(usb_devices))
    cameras = []
    captures = []
    for idx, i in enumerate(usb_devices):
        if i == target_camera:
            cameras.append(idx)
    else:
        print("Target camera " + target_camera + " ist found")
        return

    for j in cameras:
        video = cv2.VideoCapture(j, cv2.CAP_DSHOW)
        props = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_BRIGHTNESS: 94,
            cv2.CAP_PROP_AUTOFOCUS: 0,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_TRIGGER: -8,
            cv2.CAP_PROP_FOCUS: 0,
            cv2.CAP_PROP_SATURATION: 255,
            cv2.CAP_PROP_SHARPNESS: 0,
            cv2.CAP_PROP_CONTRAST: 255,
            cv2.CAP_PROP_SETTINGS:1
        }
        for key, value in props.items():
            video.set(key, value)
        success, _ = video.read()
        print("camera {} initialized {}".format(j, success))
        if not success:
            continue
        captures.append(video)

    if not len(captures) == len(cameras):
        print("you are stupid?")
    return captures


def camera_init(width = 620,high = 480):
    vid = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    # Set camera parameters
    vid.set(cv2.CAP_PROP_FPS, 60)
    vid.set(cv2.CAP_PROP_CONTRAST, 255)  # Set contrast to 0.8
    vid.set(cv2.CAP_PROP_BRIGHTNESS, 0)  # Set brightness to 0.5
    vid.set(cv2.CAP_PROP_SATURATION, 1000)  # Set saturation to 0.6
    vid.set(cv2.CAP_PROP_HUE, 1000)  # Set hue to 0.3
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Set width to 640
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, high)  # Set height to 480
    return vid, True



#   detects moving object (X, Y) and take point to not detect around it
class MovingObjectDetector:
    def __init__(self, learning_rate=0.01):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.learning_rate = learning_rate

    def detect(self, image, xy_point=None):
        # Apply background subtraction to obtain the moving objects
        fgmask = self.subtractor.apply(image, learningRate=self.learning_rate)

        # xy_point_rect = (int(xy_point[0]-rect_scale), int(xy_point[1]-rect_scale)), (int(xy_point[0])+rect_scale, int(xy_point[1])+rect_scale)

        # Apply morphology operations to remove noise and fill gaps in the moving objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the foreground mask
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour that is not within another contour and has an area > 500
        largest_contour = None
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if hierarchy[0][i][3] == -1 and area > 250:
                if largest_contour is None or area > cv2.contourArea(largest_contour):
                    if xy_point is not None:
                        rect = cv2.minAreaRect(contour)
                        if cv2.rotatedRectangleIntersection(rect, (xy_point, (225, 225), 0))[0] == cv2.INTERSECT_NONE:
                            largest_contour = contour

                        cv2.circle(image, (int(xy_point[0]), int(xy_point[1])),100, (255, 50, 50), 2)
                    else:
                        largest_contour = contour

        # Get the centroid of the bounding rectangle of the largest contour
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            centroid_x = x + w / 2
            centroid_y = y + h / 2
            return (centroid_x, centroid_y)
        else:
            return None


def load_calib_data(width, height):
    # Load the camera calibration data
    calib_data = np.load('calibration_data.npz')
    mtx = calib_data['mtx']
    dist = calib_data['dist']
    # Calculate the new camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 0, (width, height))
    return mtx, dist, newcameramtx, roi

class ImageProcessor:
    def __init__(self, window_name):
        self.window_name = window_name
        self.lower = np.array([16, 94, 102])
        self.upper = np.array([85, 255, 255])
        self.image = None
        self.mask = None
        self.hsv = None
        self.create_trackbars()

    def process_image(self, image):
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(image, -1, kernel)
        blur = cv2.blur(dst, (6, 6))
        self.image = image
        self.create_mask()
        return self.image

    def create_trackbars(self):
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar('Lower R', self.window_name, self.lower[0], 255, self.update_trackbars)
        cv2.createTrackbar('Lower G', self.window_name, self.lower[1], 255, self.update_trackbars)
        cv2.createTrackbar('Lower B', self.window_name, self.lower[2], 255, self.update_trackbars)
        cv2.createTrackbar('Upper R', self.window_name, self.upper[0], 255, self.update_trackbars)
        cv2.createTrackbar('Upper G', self.window_name, self.upper[1], 255, self.update_trackbars)
        cv2.createTrackbar('Upper B', self.window_name, self.upper[2], 255, self.update_trackbars)

    def update_trackbars(self, val):
        self.lower[0] = cv2.getTrackbarPos('Lower R', self.window_name)
        self.lower[1] = cv2.getTrackbarPos('Lower G', self.window_name)
        self.lower[2] = cv2.getTrackbarPos('Lower B', self.window_name)
        self.upper[0] = cv2.getTrackbarPos('Upper R', self.window_name)
        self.upper[1] = cv2.getTrackbarPos('Upper G', self.window_name)
        self.upper[2] = cv2.getTrackbarPos('Upper B', self.window_name)
        self.create_mask()

    def create_mask(self):
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv, self.lower, self.upper)
        return self.mask

    def show_image(self):
        cv2.imshow(self.window_name, self.mask)
        return self.mask


def find_triangle(image, target_image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Initialize an empty array to store the coordinates
    point_array = []
    if len(contours) == 0:
        # No contours found, return None for all values
        return target_image, None, None
    else:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 <= area <= 5000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.07 * peri, True)
                objCor = len(approx)

                if objCor == 3:
                    for point in approx:
                        x, y = point[0]
                        point_array.append([x, y])
                        cv2.circle(target_image, point[0], 3, (0, 0, 255), -1)
                        #
                        # cv2.putText(target_image,str(point[0]), (point[0][0] - 20, point[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        #             0.5, (255, 0, 0), 2, cv2.LINE_AA)

        if len(point_array) == 3:
            botom_point = find_closest_point(point_array)
            top_point = find_furthest_point(point_array, botom_point)

            cv2.circle(target_image, botom_point, 3, (0, 0, 255), -1)
            cv2.circle(target_image, top_point, 5, (255, 0, 0), -1)

            cv2.line(target_image, botom_point, top_point, (0, 0, 255), 2)
            cv2.putText(target_image, "Botom point", botom_point, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(target_image, "Top point", top_point, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)
            return botom_point, top_point
        else:
            return None, None


def find_closest_point(points):
    """ Returns midlepoint of shortest line of triangle
        .
      .   .
     .     .
    .___o___.
    """
    # Initialize variables for the smallest distance and the closest points
    smallest_distance = None
    closest_points = None
    # Loop over all pairs of points and find the smallest distance
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Calculate the distance between the current pair of points
            distance = math.sqrt((points[j][0] - points[i][0]) ** 2 + (points[j][1] - points[i][1]) ** 2)

            # Update the smallest_distance and closest_points if necessary
            if smallest_distance is None or distance < smallest_distance:
                smallest_distance = distance
                closest_points = (points[i], points[j])

    # Calculate the center point on the line between the closest points
    x1, y1 = closest_points[0]
    x2, y2 = closest_points[1]

    center_point = (x1 + x2) / 2, (y1 + y2) / 2
    center_point = (int(center_point[0]), int(center_point[1]))
    return center_point


def find_furthest_point(points, center_point):
    """ Returns midlepoint of shortest line of triangle
        o
      .   .
     .     .
    .       .
    """
    furthest_point = None
    furthest_distance = None

    # Loop over all points and find the furthest point
    for point in points:
        distance = math.sqrt((point[0] - center_point[0]) ** 2 + (point[1] - center_point[1]) ** 2)

        # Update the furthest_point and furthest_distance if necessary
        if furthest_distance is None or distance > furthest_distance:
            furthest_distance = distance
            furthest_point = point

    return furthest_point


def draw_min_rect(mask_image, target_image, window_name):
    """ Returns image with drawerd center point of contour and center point of minAreaRect.  """
    # Find contours in the mask image
    contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # No contours found, return None for all values
        return target_image, None, None

    # Find the contour with the maximum area
    max_area = 0
    max_contour = None
    rect_center = None
    contour_center = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area > 200:
            max_area = area
            max_contour = contour

    # Draw the minimum area rectangle around the contour on the target image
    if max_contour is not None:
        rect = cv2.minAreaRect(max_contour)
        box = np.int0(cv2.boxPoints(rect))
        rect_center = ((box[0][0] + box[2][0]) // 2, (box[0][1] + box[2][1]) // 2)
        cv2.drawContours(target_image, [box], 0, (0, 255, 0), 2)

    M = cv2.moments(max_contour)
    for cnt in contours:
        if M["m00"] != 0:
            contour_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            contour_center = np.array(contour_center)

            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Circle for contour center
            cv2.circle(target_image, contour_center, 3, (0, 0, 255), -1)
            # Circle for rect center
            cv2.circle(target_image, rect_center, 3, (0, 255, 255), -1)
            # text above rect center
            cv2.putText(target_image, window_name, (rect_center[0] - 20, rect_center[1] - 20), font,
                        0.5, (255, 0, 0), 2, cv2.LINE_AA)

        else:
            cx, cy = None, None
    return target_image, rect_center, contour_center


# ----------------MATH-------------------------

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def dot_product(p1, p2, p3):
    """ Returns 2 vectors from 3 points.  """
    if p1 is not None and p2 is not None and p3 is not None:
        # Handle the case where either p1 or p3 is None
        a = (p1[0] - p2[0], p1[1] - p2[1])
        b = (p1[0] - p3[0], p1[1] - p3[1])
        return a, b
    else:
        return (0, 0), (0, 0)


def mag(x):
    return math.sqrt(sum(i ** 2 for i in x))


def angle_between_and_direcrion(v1, v2, deg=True):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    distance = int(mag(v2))

    if deg:
        result = round(np.degrees([radians.real])[0], 1)
        x1, x2, y1, y2 = v1[0], v1[1], v2[0], v2[1]
        a = math.atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2);

        if a >= 0:
            return result, distance
        else:
            return result * -1, distance


def put_text_on_point(target_image, text, point, adjust):
    if point is not None:
        """Puts text above point"""
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # text above rect center
        cv2.putText(target_image, str(text), (point[0], point[1] - adjust), font,
                    0.5, (255, 0, 0), 2, cv2.LINE_AA)


#   PID Controller
class PIDController:
    def __init__(self, kp, ki, kd, min_output, max_output, max_integral):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.min_output = min_output
        self.max_output = max_output

        self.max_integral = max_integral  # maximum value for the I term
        self.integral = 0
        self.prev_error = 0
        self.last_time = time.monotonic()
        self.last_output = 0

    def calculate(self, setpoint, process_variable):
        current_time = time.monotonic()
        elapsed_time = current_time - self.last_time

        error = setpoint - abs(process_variable)

        # Proportional term
        P = self.Kp * error

        # Integral term
        self.integral += error * elapsed_time * self.Ki

        # Limit the I term to a maximum value
        if self.integral > self.max_integral:
            self.integral = self.max_integral
        elif self.integral < -self.max_integral:
            self.integral = -self.max_integral

        I = self.integral

        # Derivative term
        D = self.Kd * (error - self.prev_error) / elapsed_time
        self.prev_error = error

        # Compute PID output and limit to max/min values
        output = I + P + D

        if output > self.max_output:
            output = self.max_output
        elif output < self.min_output:
            output = self.min_output

        # Update last_time variable
        self.last_time = current_time
        if math.isnan(output):
            return 0
        else:
            return int(output)
