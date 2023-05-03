import serial

# establish serial connection
ser = serial.Serial('COM15', 115200, timeout=1) # replace 'COM3' with your serial port name

# send a command to the transmitter
ser.write(b'My command\r\n') # replace 'My command' with your command

while True:
    # read data from the transmitter
    data = ser.readline().decode().strip()
    print(data)

# close serial connection
ser.close()