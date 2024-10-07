#include <Servo.h>

const int servoPin = 9;

const int OPEN_POSITION = 120; 
const int CLOSED_POSITION = 0;  


const unsigned long HOLD_DURATION = 2000; 

const unsigned long BAUD_RATE = 9600;

const char PLASTIC_DETECTED_SIGNAL = '1';



Servo myServo;

void setup() {
  Serial.begin(BAUD_RATE);
  myServo.attach(servoPin);
  myServo.write(CLOSED_POSITION);
  #if defined(ARDUINO_SAM_DUE)
    while (!Serial);
  #endif
  
  Serial.println("Servo Controller Initialized.");
}

void loop() {
  if (Serial.available() > 0) {
    char incomingByte = Serial.read();
    Serial.print("Received byte: ");
    Serial.println(incomingByte);
    if (incomingByte == PLASTIC_DETECTED_SIGNAL) {
      Serial.println("Plastic detected! Activating servo...");
      myServo.write(OPEN_POSITION);
      Serial.print("Servo moved to ");
      Serial.print(OPEN_POSITION);
      Serial.println(" degrees.");
      delay(HOLD_DURATION);
      myServo.write(CLOSED_POSITION);
      Serial.print("Servo moved back to ");
      Serial.print(CLOSED_POSITION);
      Serial.println(" degrees.");
      
      Serial.println("Servo operation complete.");
    }
    else {
      Serial.print("Unknown signal received: ");
      Serial.println(incomingByte);
    }
  }
  delay(10);
}
