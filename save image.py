import cv2 as cv
import numpy as np
from PIL import Image
from math import atan
import math
from utils import *


# Load the cascade classifier
vid = camera_init()
processor = ImageProcessor("My Image")

show_image = 0
while True:
    ret, frame = vid.read()
    frame = cv.rotate(frame, cv.ROTATE_180)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # find our robot
    processor.process_image(frame)
    if not show_image:
        mask = processor.show_image()
    else:
        mask = processor.create_mask()

    image_to_draw, rect_center, contour_center = draw_min_rect(mask, frame)

    cv.imshow("target_image", image_to_draw)


    if cv.waitKey(1) == ord('q'):
        break

    if cv.waitKey(1) == ord('c') and show_image == 0:
        show_image = 1
        cv2.destroyWindow(processor.window_name)

vid.release()
cv.destroyAllWindows()