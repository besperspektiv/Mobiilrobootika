import cv2
import numpy as np

def nothing(x):
    pass

img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('mask')

cv2.createTrackbar('R', 'mask', 0, 255, nothing)
cv2.createTrackbar('G', 'mask', 0, 255, nothing)
cv2.createTrackbar('B', 'mask', 0, 255, nothing)


tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW']
tracker_type = tracker_types[4]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

vid = cv2.VideoCapture(0)
ret, frame = vid.read()

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_pink = np.array([0, 0, 0])  # BRG
upper_pink = np.array([116, 255, 255])

mask = cv2.inRange(hsv, lower_pink, upper_pink)
frame = cv2.bitwise_not(mask)

# Resize the video for a more convinient view
# frame = cv2.resize(frame, [frame_width//2, frame_height//2])

if not ret:
    print('cannot read the video')
# Select the bounding box in the first frame
bbox = cv2.selectROI(frame, False)
ret = tracker.init(frame, bbox)
# Start tracking

while True:
    ret, frame = vid.read()
    image_coppy = frame.copy()

    if not ret:
        print('something went wrong')
        break

    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(image_coppy, -1, kernel)
    blur = cv2.blur(dst, (6, 6))
    median = cv2.medianBlur(blur, 5)

    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

    R = cv2.getTrackbarPos('R', 'mask')
    G = cv2.getTrackbarPos('G', 'mask')
    B = cv2.getTrackbarPos('B', 'mask')

    lower_pink = np.array([0, 0, 0])  # BRG
    upper_pink = np.array([116, 255, 255])

    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask = cv2.bitwise_not(mask)

    timer = cv2.getTickCount()
    ret, bbox = tracker.update(mask)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, tracker_type + " Tracker", (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)




    cv2.imshow("mask", mask)
    cv2.imshow("Tracking", frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27: break
cv2.destroyAllWindows()