"""
Used to send data to arduino and back
"""

import serial
import time
import serial.tools.list_ports

startMarker = '<'
endMarker = '>'
dataStarted = False
dataBuf = ""
messageComplete = False


# ========================
# ========================
# the functions
def scan_com_ports():
    ports = serial.tools.list_ports.comports()
    if len(ports) == 0:
        print("No COM ports available.")
    else:
        for port in ports:
            description = port.description
            if "CH340" in description:
                return port.name
        print("Can't find correct COM port.")

def setupSerial(baudRate):
    serialPortName = scan_com_ports()
    global serialPort

    serialPort = serial.Serial(port=serialPortName, baudrate=baudRate, timeout=0, rtscts=True)

    print("Serial port " + serialPortName + " opened  Baudrate " + str(baudRate))

    waitForArduino()


# ========================

def sendToArduino(stringToSend):
    # this adds the start- and end-markers before sending
    global startMarker, endMarker, serialPort

    stringWithMarkers = (startMarker)
    stringWithMarkers += stringToSend
    stringWithMarkers += (endMarker)
    serialPort.write(stringWithMarkers.encode('utf-8'))  # encode needed for Python3


# ==================

def recvLikeArduino():
    global startMarker, endMarker, serialPort, dataStarted, dataBuf, messageComplete

    if serialPort.inWaiting() > 0 and messageComplete == False:
        x = serialPort.read().decode("utf-8")  # decode needed for Python3

        if dataStarted == True:
            if x != endMarker:
                dataBuf = dataBuf + x
            else:
                dataStarted = False
                messageComplete = True
        elif x == startMarker:
            dataBuf = ''
            dataStarted = True

    if (messageComplete == True):
        messageComplete = False
        return dataBuf
    else:
        return "XXX"

    # ==================


def waitForArduino():
    # wait until the Arduino sends 'Arduino is ready' - allows time for Arduino reset
    # it also ensures that any bytes left over from a previous message are discarded

    print("Waiting for Arduino to reset")

    msg = ""
    while msg.find("Arduino is ready") == -1:
        msg = recvLikeArduino()
        if not (msg == 'XXX'):
            print(msg)


def send_signal_to_motors(signal1=1500, signal2=1500, signal3=1500):
    # check for a reply
    arduinoReply = recvLikeArduino()
    if not (arduinoReply == 'XXX'):
        print("Time %s  Reply %s" % (time.time(), arduinoReply))
    sendToArduino(str(signal1) + "," + str(signal2) + "," + str(signal3))


if __name__ == '__main__':
    setupSerial(115200, "COM7")
    while True:
        send_signal_to_motors()
        prevTime = time.time()