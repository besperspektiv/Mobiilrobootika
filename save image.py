import cv2 as cv
import imutils
import numpy as np
import win32con
import win32gui
import win32ui
from PIL import Image
from math import atan
import math


"""
To train model go to this link https://amin-ahmadi.com/cascade-trainer-gui/
"""
from utils import ImageProcessor

CAMERA = True
vid = cv.VideoCapture(0, cv.CAP_DSHOW)
# Set camera parameters
vid.set(cv.CAP_PROP_FPS, 60)
vid.set(cv.CAP_PROP_CONTRAST, 250)  # Set contrast to 0.8
vid.set(cv.CAP_PROP_BRIGHTNESS, 100)  # Set brightness to 0.5
vid.set(cv.CAP_PROP_SATURATION, 250)  # Set saturation to 0.6
vid.set(cv.CAP_PROP_HUE, 250)  # Set hue to 0.3
vid.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480

while True:
    ret, im = vid.read()
    im = cv.rotate(im, cv.ROTATE_180)

    hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)



    cv.imshow("thresh1", im)
    cv.imshow("mage", gray)

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        vid.release()
        break
