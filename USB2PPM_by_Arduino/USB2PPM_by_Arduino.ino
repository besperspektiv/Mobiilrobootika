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


volatile byte _cmdbuffer[BUFFSIZE + 1];
volatile int _buffindex = 0;

volatile bool onState = default_onState;

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
    1500,  // Channel 3 default value
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

  Serial.begin(9600);
}

//////////////////////////////////////////  LOOP  ////////////////////////////////////
void loop(){
  if (Serial.available() >= 12) { // Wait until we have at least 12 bytes available
    byte buffer[12];
    Serial.readBytes(buffer, 12);

    int signal1 = *(int*)(buffer);
    int signal2 = *(int*)(buffer + 4);
    int signal3 = *(int*)(buffer + 8);

    ppmDefaultChannelValue[3] = signal1;  // Chanel 1 
    // ppmDefaultChannelValue[1] = signal1;  // Chanel 1
    // ppmDefaultChannelValue[2] = signal1;  // Chanel 1 
  }
    
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