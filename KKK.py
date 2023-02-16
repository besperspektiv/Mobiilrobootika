# echo-client.py
import socket
import time

start = time.time()
x = 0
HOST = "192.168.10.128"  # The server's hostname or IP address
PORT = 8888  # The port used by the server

def to_byte(info_to_byte):
    to_byte = bytes(str(info_to_byte), 'utf-8')
    print(info_to_byte)
    return to_byte

while True:

    if time.time() - start > 0.1:
        pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(to_byte(1))