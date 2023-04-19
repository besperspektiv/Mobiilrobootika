import cv2
import time
import numpy as np
# Open the video capture device
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set camera parameters
vid.set(cv2.CAP_PROP_FPS, 60)
vid.set(cv2.CAP_PROP_CONTRAST, 255)  # Set contrast to 0.8
vid.set(cv2.CAP_PROP_BRIGHTNESS, 200)  # Set brightness to 0.5
vid.set(cv2.CAP_PROP_SATURATION, 1000)  # Set saturation to 0.6
vid.set(cv2.CAP_PROP_HUE, 1000)  # Set hue to 0.3
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Initialize variables for calculating fps
num_frames = 0
start_time = time.time()

# Get the video width and height
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Choose the codec
fps = 10.0 # Choose the desired frame rate
width = int(frame_width) # Get the width of the frames
height = int(frame_height) # Get the height of the frames
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while True:
    # Read a frame from the video capture device
    ret, frame = vid.read()
    clear_frame = frame.copy()
    if not ret:
        break

    # Increment the number of frames
    num_frames += 1

    # Display the frame in a window called "Video Feed"
    # Calculate the elapsed time since the start of the program
    elapsed_time = time.time() - start_time

    # Calculate the current fps
    current_fps = num_frames / elapsed_time

    # Display the current fps
    clear_frame = cv2.putText(clear_frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(current_fps)
    # Write the frame to the output video file
    out.write(clear_frame)
    cv2.imshow("Video Feed", frame)
    cv2.imshow("Video clear_frame", clear_frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture device and close all windows
vid.release()
cv2.destroyAllWindows()
