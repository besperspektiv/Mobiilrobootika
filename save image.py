import cv2 as cv
import numpy as np
import win32con
import win32gui
import win32ui
from PIL import Image
from math import atan
import math

import robot_motion


CAMERA = True
vid = cv.VideoCapture(0)


def nothing(x):
    pass

img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('mask')

cv.createTrackbar('R', 'mask', 0, 255, nothing)
cv.createTrackbar('G', 'mask', 0, 255, nothing)
cv.createTrackbar('B', 'mask', 0, 255, nothing)


def list_window_names():
    """ Prints list of windows.  """
    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            print(hex(hwnd), win32gui.GetWindowText(hwnd))
    win32gui.EnumWindows(winEnumHandler, None)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def dot_product(p1, p2, p3):
    """ Returns 2 vectors from 3 points.  """
    a = (p1[0] - p2[0], p1[1] - p2[1])
    b = (p1[0] - p3[0], p1[1] - p3[1])
    return a,b

def mag(x):
    return math.sqrt(sum(i**2 for i in x))

def angle_between_and_direcrion(v1, v2, deg=True):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    result = radians
    distance = int(mag(v2))

    if deg:
        result = round(np.degrees([radians.real])[0],1)
        x1, x2, y1, y2 = v1[0], v1[1], v2[0], v2[1]
        a = math.atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2);

        if a >= 0:
            return result, distance
        else:
            return result * -1, distance




list_window_names()
hwnd_target = win32gui.FindWindow(None, 'Preview of Asteroids - GDevelop Example') # used for test
font = cv.FONT_HERSHEY_COMPLEX

def find_contours_enemy(image, target_image):
    frame_contours = image.copy()
    # smoothen image

    gray = cv.cvtColor(frame_contours, cv.COLOR_BGR2GRAY)
    median = cv.medianBlur(frame_contours, 5)

    hsv = cv.cvtColor(median, cv.COLOR_BGR2HSV)

    lower_blue = np.array([0, 0, 73])
    upper_blue = np.array([255, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    mask = cv.bitwise_not(mask)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(image, image, mask=mask)
    ret, thresh = cv.threshold(mask, 122, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Getting the biggest contour
    if len(contours) != 0:
        for c in contours:
            # draw in blue the contours that were founded
            # cv.drawContours(target_image, contours, -1, 255, 3)

            # find the biggest countour (c) by the area
            c = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(c)
            # draw the biggest contour (c) in green
            cv.rectangle(target_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv.circle(target_image, (int(x + w/2), int(y + h/2)), 10, (255, 255, 255), -1)
            cv.putText(target_image, 'enemy', (int(x + w/2)-30, int(y + h/2)-75),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.75, (255, 0, 0), 2, cv.LINE_AA)
            return (int(x + w/2), int(y + h/2))

def find_contours(image, target_image, tresh=0, moments = 1):
    try:
        if tresh == 0:
            ret, thresh = cv.threshold(image, 100, 255, cv.THRESH_BINARY_INV)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Getting the biggest contour

        cnt = contours[0]
        M = cv.moments(cnt)

        for cnt in contours:
            if M["m00"] != 0 and moments:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                c = max(contours, key=cv.contourArea)
                rect = cv.minAreaRect(c)
                box = cv.boxPoints(rect)
                box = np.int0(box)

                # pints of contour and minAreaRect cnter points
                cv.circle(target_image, (cx, cy), 2, (0, 255, 255), -1)
                cv.putText(target_image, 'robot', (int(rect[0][0])-30, int(rect[0][1])-75), cv.FONT_HERSHEY_SIMPLEX,
                                    0.75, (255, 0, 0), 2, cv.LINE_AA)

                cv.circle(target_image, (int(rect[0][0]), int(rect[0][1])), 2, (255, 0, 255), -1)
                # line between contour and minAreaRect center point
                # cv.line(target_image, (int(rect[0][0]), int(rect[0][1])), (cx, cy), (0, 0, 0), 2)
                cv.drawContours(target_image, [box], 0, (0, 0, 255), 2)

                try:
                    return (int(rect[0][0]), int(rect[0][1])), (cx, cy)
                except:
                    return (0, 0), (0, 0)

    except:
        print("No contour found")

    else:
        cx, cy = 0, 0

    if len(contours) != 0:
        # draw in blue the contours that were founded
        #cv.drawContours(image_coppy, contours, -1, 255, 3)
        # find the biggest countour (c) by the area
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        # draw the biggest contour (c) in green
        # cv.rectangle(target_image, (x, y), (x + w, y + h), (255, 255, 0), 2)





while True:
    if CAMERA == False:
        """Crop window image"""
        try:
            left, top, right, bot = win32gui.GetWindowRect(hwnd_target)
            w = right - left
            h = bot - top

            win32gui.SetForegroundWindow(hwnd_target)

            hdesktop = win32gui.GetDesktopWindow()
            hwndDC = win32gui.GetWindowDC(hdesktop)
            mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
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
        except:
            robot_motion.write_force_to_file(0, 0)
            print('Force 0, 0')
            print("Select right window!")
    else:
        result = 0
        ret, im = vid.read()

    if result is None or ret == True:


        if not CAMERA:
            open_cv_image = np.array(im)
            cropped_image = open_cv_image[30:650, 8:805]
            image_coppy = cropped_image.copy()
        else:
            cropped_image = im
            image_coppy = im.copy()



        gray = cv.cvtColor(image_coppy, cv.COLOR_BGR2GRAY)
        grey_3_channel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv.filter2D(grey_3_channel, -1, kernel)
        blur = cv.blur(dst, (6, 6))
        median = cv.medianBlur(blur, 5)
        hsv = cv.cvtColor(median, cv.COLOR_BGR2HSV)


        dst = cv.filter2D(image_coppy, -1, kernel)
        blur = cv.blur(dst, (6, 6))
        median = cv.medianBlur(blur, 5)

        hsv = cv.cvtColor(median, cv.COLOR_BGR2HSV)

        R = cv.getTrackbarPos('R', 'mask')
        G = cv.getTrackbarPos('G', 'mask')
        B = cv.getTrackbarPos('B', 'mask')

        lower_pink = np.array([0, 0, 0]) # BRG
        upper_pink = np.array([R, G, B])

        mask = cv.inRange(hsv, lower_pink, upper_pink)
        mask = cv.bitwise_not(mask)

        # Bitwise-AND mask and original image
        res = cv.bitwise_and(cropped_image, cropped_image, mask=mask)


        gray1 = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

        try:
            ourXY, ourXY_center = find_contours(gray1, image_coppy,1,1)
        except:
            print("FUUCK")
        # enemyXY = find_contours_enemy(image_coppy, image_coppy)
        #
        # a,b = dot_product(ourXY_center, ourXY, enemyXY)
        # angle,distance = angle_between_and_direcrion(a,b)
        #
        # robot_motion.robot_motion(angle, distance)
        #
        # cv.line(image_coppy, ourXY, enemyXY, (0, 255, 0), 2)
        cv.imshow('mask', mask)
        cv.imshow('Computer', image_coppy)


        # press 'q' with the output window focused to exit.
        # waits 1 ms every loop to process key presses
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            cap.release()
            out.release()
            break