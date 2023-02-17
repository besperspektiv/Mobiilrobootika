import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, window_name):
        self.image = None
        self.window_name = window_name

    def process_image(self, image):
        self.image = image.copy()
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(self.image, -1, kernel)
        blur = cv2.blur(dst, (6, 6))
        self.image = cv2.medianBlur(blur, 5)

        # Implement image processing on self.image
        pass

    def create_trackbars(self):
        def nothing(x):
            pass

        window_name = str(self.window_name)
        cv2.namedWindow(window_name)
        cv2.createTrackbar('Lower_R', self.window_name, 0, 255, nothing)
        cv2.createTrackbar('Lower_G', self.window_name, 0, 255, nothing)
        cv2.createTrackbar('Lower_B', self.window_name, 0, 255, nothing)
        cv2.createTrackbar('Upper_R', self.window_name, 255, 255, nothing)
        cv2.createTrackbar('Upper_G', self.window_name, 255, 255, nothing)
        cv2.createTrackbar('Upper_B', self.window_name, 255, 255, nothing)

    def create_mask(self):
        lower = (cv2.getTrackbarPos('Lower_R', self.window_name),
                 cv2.getTrackbarPos('Lower_G', self.window_name),
                 cv2.getTrackbarPos('Lower_B', self.window_name))
        upper = (cv2.getTrackbarPos('Upper_R', self.window_name),
                 cv2.getTrackbarPos('Upper_G', self.window_name),
                 cv2.getTrackbarPos('Upper_B', self.window_name))
        mask = cv2.inRange(self.image, lower, upper)
        return mask

    def show_image(self):
        if self.image is not None:
            masked_image = self.image.copy()
            mask = self.create_mask()
            cv2.imshow(self.window_name, mask)
