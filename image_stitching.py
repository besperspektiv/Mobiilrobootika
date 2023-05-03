import cv2

# Initialize the cameras
cam1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

widh, hight = 640,380
# Set the resolution for both cameras
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, widh)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, hight)
cam1.set(cv2.CAP_PROP_BRIGHTNESS, 100)  # Set brightness to 0.5
cam1.set(cv2.CAP_PROP_FPS, 30)

cam2.set(cv2.CAP_PROP_FRAME_WIDTH, widh)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, hight)
cam2.set(cv2.CAP_PROP_BRIGHTNESS, 100)  # Set brightness to 0.5
cam2.set(cv2.CAP_PROP_FPS, 30)

# Capture the first frame from each camera
_, frame1 = cam1.read()
_, frame2 = cam2.read()

# Stitch the two frames together
stitcher = cv2.createStitcher() if cv2.__version__.startswith('3') else cv2.Stitcher.create()
status, stitched_image = stitcher.stitch([frame1, frame2])

# If stitching was successful, display the result
if status == cv2.STITCHER_OK:
    cv2.imshow("Panorama", stitched_image)
    cv2.waitKey(0)

# Loop to capture and stitch frames from the two cameras
while True:
    # Capture a frame from each camera
    _, frame1 = cam1.read()
    _, frame2 = cam2.read()

    # Stitch the two frames together
    status, stitched_image = stitcher.stitch([frame1, frame2])

    # If stitching was successful, display the result
    if status == cv2.STITCHER_OK:
        # Apply adjustments to the panorama image here
        # For example, you can resize the image, adjust brightness/contrast, etc.
        stitched_image = cv2.resize(stitched_image, (640, 480))
        stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Panorama", stitched_image)

    # Check for key press to exit
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the cameras and destroy all windows
cam1.release()
cam2.release()
cv2.destroyAllWindows()
