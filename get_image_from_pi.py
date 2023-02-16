import socket
import cv2
import numpy as np

client_socket = socket.socket()
client_socket.connect(("192.168.10.119", 8000))

while True:
    received_data = b""
    while True:
        new_data = client_socket.recv(1024)
        received_data += new_data
        if len(new_data) < 1024:
            break

    nparr = np.frombuffer(received_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

client_socket.close()
cv2.destroyAllWindows()

