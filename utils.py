import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, window_name):
        self.window_name = window_name
        self.lower = np.array([0, 0, 0])
        self.upper = np.array([255, 255, 255])
        self.image = None
        self.mask = None
        self.create_trackbars()

    def process_image(self, image):
        self.image = image.copy()
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(self.image, -1, kernel)
        blur = cv2.blur(dst, (6, 6))
        self.image = cv2.medianBlur(blur, 5)
        self.create_mask()

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
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(hsv, self.lower, self.upper)

    def show_image(self):
        cv2.imshow(self.window_name, self.mask)
        return self.mask

