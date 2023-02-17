import cv2 as cv
import numpy as np
import win32con
import win32gui
import win32ui
from PIL import Image
from math import atan
import math


from utils import ImageProcessor

CAMERA = True
vid = cv.VideoCapture(0, cv.CAP_DSHOW)

processor = ImageProcessor("enemy")
processor.create_trackbars()

processor1 = ImageProcessor("robot")
processor1.create_trackbars()

while True:
    ret, im = vid.read()

    processor.process_image(im)
    processor.show_image()

    processor1.process_image(im)
    processor1.show_image()


    if cv.waitKey(1) == ord('c'):
        cv.destroyAllWindows()
        break

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        vid.release()
        break