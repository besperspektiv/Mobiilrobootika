/*
#include <SoftwareSerial.h>

#define HEADER_SIZE 4

SoftwareSerial mySerial(10, 11); // RX, TX

void setup() {
  Serial.begin(9600);
  mySerial.begin(9600);
}

void loop() {
  if (Serial.available() >= HEADER_SIZE) {
    // Read the packet header
    uint16_t signal1, signal2;
    Serial.readBytes((char*)&signal1, 2);
    Serial.readBytes((char*)&signal2, 2);

    // Process the payload
    mySerial.print("Signal 1: ");
    mySerial.println(signal1);
    mySerial.print("Signal 2: ");
    mySerial.println(signal2);
  }
  
  if (mySerial.available()) {
    char c = mySerial.read();
    if (c == 0x06) {
      mySerial.println("Acknowledgement received");
    }
  }
}
*/