import cv2
from capture_window import*
from utils import ImageProcessor
list_window_names()



processor = ImageProcessor("My Image")


# Create Shi-Tomasi detector
feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)

# Create Lucas-Kanade optical flow algorithm
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables
old_gray = None
old_points = None

configured = False

frame = crop_image_from_window('Preview of Asteroids - GDevelop Example')
while not configured:
    processor.process_image(frame)
    processor.show_image()

    if cv2.waitKey(1) == ord('c'):
        configured = True
        cv2.destroyWindow(processor.window_name)
else:
    while True:
        # Capture frame-by-frame
        frame = crop_image_from_window('Preview of Asteroids - GDevelop Example')
        processor.process_image(frame)
        mask = processor.show_image()

        if old_gray is not None and old_points is not None:
            # Calculate optical flow using Lucas-Kanade algorithm
            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, mask, old_points, None, **lk_params)

            if new_points is not None:
                # Select good points
                good_new = new_points[status == 1]
                good_old = old_points[status == 1]

                # Draw tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)
            else:
                print("No points found")

            # Update variables
            old_gray = mask.copy()
            old_points = good_new.reshape(-1, 1, 2)
        else:
            # Detect features in first frame
            old_gray = mask.copy()
            old_points = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release the capture and destroy all windows
    cv2.destroyAllWindows()