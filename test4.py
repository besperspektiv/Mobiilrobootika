import cv2
import numpy as np
from time import time, sleep

captures = []

# ls /dev/v4l/by-path/
cameras = [
1,
3,
]

for j in cameras:
    video = cv2.VideoCapture(j, cv2.CAP_DSHOW)
    video.set(3, 640)
    video.set(4, 480)
    success, _ = video.read()
    print("camera {} initialized {}".format(j, success))
    if not success:
        continue
    captures.append(video)

if not len(captures) == len(cameras):
    print("you are stupid?")

count, last = 0, time()
while True:
    now = time()
    if now - last > 2:
        print("%d fps" % (count * 1.0 / (now - last)))
        last = now
        count = 0
    count += 1

    frames = [cap.read()[1] for cap in captures]
    frames = [cv2.resize(frame, (0, 0), fx=0.3, fy=0.3) for frame in frames]

    #stackedA = np.hstack(frames[:4])
    #stackedB = np.hstack(frames[4:])
    #stacked = np.vstack([stackedA, stackedB])
    cv2.imshow('just a name', np.hstack(frames))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break