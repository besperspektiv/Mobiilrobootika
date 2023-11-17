import cv2
import numpy as np
import win32con
import win32gui
import win32ui
from PIL import Image
import math


def camera_init():
    vid = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # Set camera parameters
    vid.set(cv2.CAP_PROP_FPS, 60)
    vid.set(cv2.CAP_PROP_CONTRAST, 255)  # Set contrast to 0.8
    vid.set(cv2.CAP_PROP_BRIGHTNESS, 0)  # Set brightness to 0.5
    vid.set(cv2.CAP_PROP_SATURATION, 1000)  # Set saturation to 0.6
    vid.set(cv2.CAP_PROP_HUE, 1000)  # Set hue to 0.3
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480
    return vid


def load_calib_data():
    # Load the calibration data
    calib_data = np.load('calibration_data.npz')
    mtx = calib_data['mtx']
    dist = calib_data['dist']
    # Calculate the new camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 0, (640, 480))
    return mtx, dist, newcameramtx, roi


def show(frame, mtx, dist, newcameramtx, roi):
    # Undistort the frame using the calibration data
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    frame = frame[y:y + h, x:x + w]

    return frame


def list_window_names():
    """ Prints list of windows.  """

    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            print(hex(hwnd), win32gui.GetWindowText(hwnd))

    win32gui.EnumWindows(winEnumHandler, None)


def crop_image_from_window(window_name):
    hwnd_target = win32gui.FindWindow(None, window_name)
    """Crop window image"""
    try:
        left, top, right, bot = win32gui.GetWindowRect(hwnd_target)
        w = right - left
        h = bot - top

        win32gui.SetForegroundWindow(hwnd_target)

        hdesktop = win32gui.GetDesktopWindow()
        hwndDC = win32gui.GetWindowDC(hdesktop)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

        saveDC.SelectObject(saveBitMap)

        result = saveDC.BitBlt((0, 0), (w, h), mfcDC, (left, top), win32con.SRCCOPY)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hdesktop, hwndDC)

        if result is None:
            open_cv_image = np.array(im)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            return open_cv_image
    except:
        img = np.zeros((300, 512, 3), np.uint8)
        print("Select right window!")
        return img


class ImageProcessor:
    def __init__(self, window_name, enemy=0):
        self.window_name = window_name
        if not enemy:
            self.lower = np.array([0, 0, 237])
            self.upper = np.array([207, 255, 255])
        else:
            self.lower = np.array([0, 49, 202])
            self.upper = np.array([255, 255, 255])
        self.image = None
        self.mask = None
        self.hsv = None
        self.create_trackbars()

    def process_image(self, image):
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(image, -1, kernel)
        blur = cv2.blur(dst, (6, 6))
        self.image = cv2.medianBlur(blur, 5)
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
    result = radians
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
