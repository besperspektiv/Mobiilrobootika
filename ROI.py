import cv2
import time

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set camera parameters
vid.set(cv2.CAP_PROP_FPS, 60)
vid.set(cv2.CAP_PROP_CONTRAST, 255)  # Set contrast to 0.8
vid.set(cv2.CAP_PROP_BRIGHTNESS, 200)  # Set brightness to 0.5
vid.set(cv2.CAP_PROP_SATURATION, 1000)  # Set saturation to 0.6
vid.set(cv2.CAP_PROP_HUE, 1000)  # Set hue to 0.3
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)

# Read the first frame
ret, frame = vid.read()
if not ret:
    print('Error: cannot read video file')
    exit()

# Select the bounding box of the object to track
bbox = cv2.selectROI(frame, False)

# Initialize the tracker
tracker = cv2.TrackerMIL_create()
tracker.init(frame, bbox)


num_frames = 0
start_time = time.time()
while True:
    # Read the next frame
    ret, frame = vid.read()
    if not ret:
        # End of video
        break

    # Update the tracker
    ok, bbox = tracker.update(frame)
    if ok:
        # Draw the bounding box around the object
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
    else:
        # Object lost, do something
        pass

    # Increment the number of frames
    num_frames += 1
    elapsed_time = time.time() - start_time
    current_fps = num_frames / elapsed_time
    clear_frame = cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Tracking', frame)

    # Wait for key press to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close all windows
vid.release()
cv2.destroyAllWindows()