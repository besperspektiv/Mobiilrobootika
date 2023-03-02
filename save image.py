import cv2 as cv
import numpy as np
from PIL import Image
from math import atan
import math
from utils import ImageProcessor

"""
To train model go to this link https://amin-ahmadi.com/cascade-trainer-gui/
"""


CAMERA = True
vid = cv.VideoCapture(1, cv.CAP_DSHOW)
# Set camera parameters
vid.set(cv.CAP_PROP_FPS, 60)
vid.set(cv.CAP_PROP_CONTRAST, 255)  # Set contrast to 0.8
vid.set(cv.CAP_PROP_BRIGHTNESS, 0)  # Set brightness to 0.5
vid.set(cv.CAP_PROP_SATURATION, 1000)  # Set saturation to 0.6
vid.set(cv.CAP_PROP_HUE, 1000)  # Set hue to 0.3
vid.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480



# processor = ImageProcessor("My Image")

while True:
    ret, frame = vid.read()
    frame = cv.rotate(frame, cv.ROTATE_180)

    hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    blurred = cv.GaussianBlur(hsv, (5, 5), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    # processor.process_image(frame)
    # processor.show_image()

    cv.imshow("image", thresh)
    # cv.imshow("tresh", thresh)



    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        vid.release()
        break
