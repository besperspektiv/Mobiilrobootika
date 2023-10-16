import socket
import struct
import random

host = "127.0.0.1"  # Change this to your Unity's IP address
port = 12345  # Use the same port as defined in Unity

def send_data_to_unity(values = (0,0)):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        data = struct.pack('ff', *values)
        sock.sendto(data, (host, port))
