import serial.tools.list_ports

def scan_com_ports():
    ports = serial.tools.list_ports.comports()
    if len(ports) == 0:
        print("No COM ports available.")
    else:
        for port in ports:
            description = port.description
        if description == "USB-SERIAL CH340 (COM17)":
            return port.name
        else:
            print("Cant find correct COM port")