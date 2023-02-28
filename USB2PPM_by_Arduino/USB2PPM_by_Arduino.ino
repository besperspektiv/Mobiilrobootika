////////////////////// APP FUNCTION //////////////////////////////
// This programm will put out a PPM signal @ sigPin (default = pin 8)


////////////////////// INCLUDES //////////////////////////////////
#include "protocol.h"

///////////////////// SYSTEM INFORMATION /////////////////////////
#define productID ARD_PPM_ENCODER // identify product 
#define fwVersion "1.0"           // firmware version

///////////////////// PARAMETERS AND SETTINGS ////////////////////
#define PPM_MinLength 1000 // minimal pulse length in µs
#define PPM_MaxLength 2000 // maximum pulse length in µs
#define PPM_FrLen 22500  //set the PPM frame length in microseconds (1ms = 1000µs)
#define PPM_PulseLen 300//set the pulse length
#define sigPin 8  //set PPM signal output pin on the arduino
#define default_channel_number 4  //set the number of chanels
#define default_onState HIGH  //set polarity of the pulses: HIGH is positive, LOW is negative
#define BUFFSIZE 24

// Variables to hold motor speed
int variable1 = 1500;
int variable2 = 1500;

///////////////////// COMMANDS ///////////////////////////////////
typedef enum { start, attention, command, stream, timeout } state;
typedef enum { idle, serial_control_write, serial_control_read, 
        define_startup_val, get_parameter, set_parameter, enable_channel, 
        disable_channel, system_info_read } command_type;


volatile byte _cmdbuffer[BUFFSIZE + 1];
volatile int _buffindex = 0;

volatile bool onState = default_onState;

volatile int _mode = attention;
volatile int _commandtype = idle;

volatile int _channel = 0; 
volatile int _position = 0;
volatile int channel_number = default_channel_number;
volatile int ppm[default_channel_number]; 

// this array holds the servo values for the ppm signal and are used as defaults
// change these values here if you need different values (usually servo values move between 1000 and 2000)

int ppmDefaultChannelValue[default_channel_number] = {
    1500,  // Channel 1 default value
    1500,  // Channel 2 default value
    1500,  // Channel 3 default value
    1500,  // Channel 4 default value
};

void setup(){  
  //initiallize default ppm values
  for(int i=0; i<channel_number; i++){
    ppm[i]= verifyConstraints(ppmDefaultChannelValue[i]);
  }

  pinMode(sigPin, OUTPUT);
  digitalWrite(sigPin, !onState);  //set the PPM signal pin to the default state (off)
  
  cli();
  TCCR1A = 0; // set entire TCCR1 register to 0
  TCCR1B = 0;
  
  OCR1A = 100;  // compare match register, change this
  TCCR1B |= (1 << WGM12);  // turn on CTC mode
  TCCR1B |= (1 << CS11);  // 8 prescaler: 0,5 microseconds at 16mhz
  TIMSK1 |= (1 << OCIE1A); // enable timer compare interrupt
  sei();

  Serial.begin(115200);
}


int a = 1000;

//////////////////////////////////////////  LOOP  ////////////////////////////////////
void loop(){
  if (Serial.available() > 0) { // check if there is serial data available to read
    String data = Serial.readStringUntil('\n'); // read the data until a newline character is received
    int commaIndex = data.indexOf(','); // find the index of the comma separator
    if (commaIndex != -1 && data.length() > commaIndex + 1) { // check if the comma separator exists and there is data after it
      int value1 = data.substring(0, commaIndex).toInt(); // extract the value for variable1
      int value2 = data.substring(commaIndex + 1).toInt(); // extract the value for variable2
      // constrain the values to be within the range of 1000 to 2000
      variable1 = constrain(value1, 1000, 2000);
      variable2 = constrain(value2, 1000, 2000);
    }
  }
  
  ppmDefaultChannelValue[3] = variable1;
  ppmDefaultChannelValue[1] = variable2;
  for(int i=0; i<channel_number; i++){
    ppm[i]= verifyConstraints(ppmDefaultChannelValue[i]);
  }
}
    


int verifyConstraints(int PPMValue) {
  if (PPMValue > PPM_MaxLength) {
    return PPM_MaxLength;
  }
  if (PPMValue < PPM_MinLength) {
    return PPM_MinLength;
  }
  return PPMValue;
}

ISR(TIMER1_COMPA_vect){  //leave this alone
  static boolean state = true;
  
  TCNT1 = 0;
  
  if(state) {  //start pulse
    digitalWrite(sigPin, onState);
    OCR1A = PPM_PulseLen * 2;
    state = false;
  }
  else{  //end pulse and calculate when to start the next pulse
    static byte cur_chan_numb;
    static unsigned int calc_rest;
  
    digitalWrite(sigPin, !onState);
    state = true;

    if(cur_chan_numb >= channel_number){
      cur_chan_numb = 0;
      calc_rest = calc_rest + PPM_PulseLen;// 
      OCR1A = (PPM_FrLen - calc_rest) * 2;
      calc_rest = 0;
    }
    else{
      OCR1A = (ppm[cur_chan_numb] - PPM_PulseLen) * 2;
      calc_rest = calc_rest + ppm[cur_chan_numb];
      cur_chan_numb++;
    }     
  }
}
