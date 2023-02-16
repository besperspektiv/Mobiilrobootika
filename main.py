from __future__ import print_function
from __future__ import division
import numpy as np
import cv2 as cv
import argparse


def nothing(x):
    pass
windowName = 'image'
cv.namedWindow(windowName)

cv.createTrackbar('r', windowName, 0, 255, nothing)
cv.createTrackbar('g', windowName, 0, 255, nothing)
cv.createTrackbar('b', windowName, 0, 255, nothing)

cv.createTrackbar('r1', windowName, 0, 255, nothing)
cv.createTrackbar('g1', windowName, 0, 255, nothing)
cv.createTrackbar('b1', windowName, 0, 255, nothing)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    img = cv.imread("image.PNG")
    frame_contours = img.copy()

    # smoothen image

    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv.filter2D(frame_contours, -1, kernel)
    blur = cv.blur(dst, (6, 6))
    median = cv.medianBlur(frame_contours, 5)

    hsv = cv.cvtColor(median, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    r = cv.getTrackbarPos('r', windowName)
    g = cv.getTrackbarPos('g', windowName)
    b = cv.getTrackbarPos('b', windowName)

    r1 = cv.getTrackbarPos('r1', windowName)
    g1 = cv.getTrackbarPos('g1', windowName)
    b1 = cv.getTrackbarPos('b1', windowName)

    lower_blue = np.array([r, g, b])
    upper_blue = np.array([r1, g1, b1])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    mask = cv.bitwise_not(mask)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(img, img, mask=mask)

    gray = cv.cvtColor(hsv, cv.COLOR_BGR2GRAY)
    grey_3_channel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)



    ret, thresh = cv.threshold(mask, 122, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Getting the biggest contour
    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv.drawContours(frame_contours, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        # draw the biggest contour (c) in green
        cv.rectangle(frame_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame

    numpy_horizontal = np.hstack((hsv, frame_contours))

    cv.imshow('frame', numpy_horizontal)
    cv.imshow(windowName, mask)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()