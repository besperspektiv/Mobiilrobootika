import cv2
import numpy as np
import win32con
import win32gui
import win32ui
from PIL import Image
import time


def configure_camera(vid):
    # Set camera parameters
    vid.set(cv2.CAP_PROP_FPS, 60)
    vid.set(cv2.CAP_PROP_CONTRAST, 255)  # Set contrast to 0.8
    vid.set(cv2.CAP_PROP_BRIGHTNESS, 255)  # Set brightness to 0.5
    vid.set(cv2.CAP_PROP_SATURATION, 0)  # Set saturation to 0.6
    vid.set(cv2.CAP_PROP_HUE, 0)  # Set hue to 0.3
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set width to 640


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
    def __init__(self, window_name):
        self.window_name = window_name
        self.lower = np.array([0, 0, 0])
        self.upper = np.array([255, 255, 255])
        self.image = None
        self.mask = None
        self.contours = None
        self.trackbar_flag = False
    def call_trackbars(self):
        if not self.trackbar_flag:
            self.trackbar_flag = True
            self.create_trackbars()

    def process_image(self, image):
        self.call_trackbars()
        self.image = image.copy()
        kernel = np.ones((5, 5), np.uint8)
        dst = cv2.filter2D(self.image, -1, kernel)
        blur = cv2.blur(dst, (6, 6))
        self.image = cv2.medianBlur(blur, 5)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.update_mask()

    def clear_image(self, image):
        self.call_trackbars()
        self.image = image.copy()
        kernel = np.ones((5, 5), np.uint8)
        dst = cv2.filter2D(self.image, -1, kernel)
        blur = cv2.blur(dst, (6, 6))
        self.image = cv2.medianBlur(blur, 5)
        self.update_mask_kernel()

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

    def update_mask(self):
        self.mask = cv2.inRange(self.image, self.lower, self.upper)

    def update_mask_kernel(self):
        self.mask = cv2.inRange(self.image, self.lower, self.upper)
        kernel = np.ones((8, 8), np.uint8)
        self.mask = cv2.erode(self.mask, kernel)


    def show_image(self):
        cv2.imshow(self.window_name, self.mask)

#   PID Controller
class PIDController:
    def __init__(self, kp, ki, kd, min_output, max_output, loop_interval):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.loop_interval = loop_interval

        self.integral = 0
        self.prev_error = 0
        self.last_time = time.monotonic()
        self.last_output = 0

    def calculate(self, setpoint, process_variable):
        current_time = time.monotonic()
        elapsed_time = current_time - self.last_time

        if elapsed_time < self.loop_interval:
            return self.last_output

        error = setpoint - process_variable

        # Proportional term
        P = self.Kp * error

        # Integral term
        self.integral += error * elapsed_time
        I = self.Ki * self.integral

        # Derivative term
        D = self.Kd * (error - self.prev_error) / elapsed_time
        self.prev_error = error

        # Compute PID output and limit to max/min values
        output = P + I + D
        if output > self.max_output:
            output = self.max_output
        elif output < self.min_output:
            output = self.min_output

        # Update last_time variable
        self.last_time = current_time
        self.last_output = output
        return output
