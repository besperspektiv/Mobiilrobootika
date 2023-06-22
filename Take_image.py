"""
Use this code to collect calibrating images for camera. camera have fisheye effect and chessboard images are used to reduce this effect.
"""
import cv2
import os
from utils import camera_init
cap = camera_init()
counter = 0
while True:
    # Capture an image
    ret, frame = cap.read()

    # Display the image
    cv2.imshow("Captured Image", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    elif key == ord('c'):
        # Create a filename for the image
        print("image saved!")
        filename = "checkerboard_images/image_{}.jpg".format(len(os.listdir())+counter)
        counter= counter+1
        print(counter)
        # Save the image
        cv2.imwrite(filename, frame)
