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


# Load the cascade classifier
face_cascade = cv.CascadeClassifier('C:\Users\pikka\PycharmProjects\Mobiilrobootika\data\classifier\cascade.xml')


# processor = ImageProcessor("My Image")

while True:
    ret, frame = vid.read()
    frame = cv.rotate(frame, cv.ROTATE_180)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # processor.process_image(frame)
    # processor.show_image()

    # Detect faces in the image using the cascade c lassifier
    robot = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in robot:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("image", frame)
    # cv.imshow("tresh", thresh)



    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        vid.release()
        break
