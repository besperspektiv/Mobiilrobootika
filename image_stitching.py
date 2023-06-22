"""
Have to be improved or replaced
not in use right now
"""
import numpy as np
import cv2
from utils import index_cameras
import time

captures = index_cameras(width=960, height=720)

def update_stitching_data(frame1, frame2):
    # create ORB feature detector
    orb = cv2.ORB_create()

    # detect keypoints and descriptors in the two frames
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    # match the keypoints in the two frames
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # find the homography matrix between the two images
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # stitch the two images together
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    dst_pts = cv2.perspectiveTransform(pts1, M)
    pts = np.concatenate((pts2, dst_pts), axis=0)
    [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-x_min, -y_min]
    H = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    return H, M, x_max, x_min, y_max, y_min, t, h2, w2


# Capture the first frame from both cameras
ret1, frame1 = captures[0].read()
ret2, frame2 = captures[1].read()

H, M, x_max, x_min, y_max, y_min, t, h2, w2 = update_stitching_data(frame1, frame2)
# image stitching function
def stitch_images(frame1, frame2, H, M, x_max, x_min, y_max, y_min, t, h2, w2):
    warped_img = cv2.warpPerspective(frame1, H.dot(M), (x_max - x_min, y_max - y_min))
    warped_img[t[1]:h2 + t[1], t[0]:w2 + t[0]] = frame2
    return warped_img

timing = time.time()
# Repeat the process in a while loop
while True:

    # Capture the frames from both cameras
    ret1, frame1 = captures[0].read()
    ret2, frame2 = captures[1].read()

    # if time.time() - timing > 5:
    #     timing = time.time()
    #     H, M, x_max, x_min, y_max, y_min, t, h2, w2 = update_stitching_data(frame1, frame2)

    # stitch the frames together
    warped_img = stitch_images(frame1, frame2,H, M, x_max, x_min, y_max, y_min, t, h2, w2)

    # show the stitched image
    cv2.imshow("Stitched Image", warped_img)

    # check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the resources
captures[0].release()
captures[1].release()
cv2.destroyAllWindows
