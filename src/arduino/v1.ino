#include <Stepper.h>
#include <NewPing.h>
#include <cvzone.h>

const byte triggerPin = 12;
const byte echoPin = 13;
SerialData serialData(1,1);
int valsRec[1];
//bool test = 1;

const int stepsPerRevolution = 200;  // Adjust for your motor
Stepper myStepper(stepsPerRevolution, 8, 10, 9, 11);

//Define maximum distance (in cm) to ping, which prevents incorrect readings
#define MAX_DISTANCE 200  

// Initialize the NewPing library
NewPing sonar(triggerPin, echoPin, MAX_DISTANCE);

void setup() {
  myStepper.setSpeed(60);
  Serial.begin(9600);
  serialData.begin();
  pinMode(7, OUTPUT);
}

void loop() {
  int distance = sonar.ping_cm();  // Measure distance in cm

  serialData.Get(valsRec);
  digitalWrite(7, valsRec[0]);

  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  if (valsRec[0] > 0) {
    if (distance > 0 && distance <= 5) {  // Check for valid readings within 5 cm
      Serial.println("Motor turning clockwise");
      myStepper.step(stepsPerRevolution);
    } else {
      Serial.println("Motor turning counterclockwise");
      myStepper.step(-stepsPerRevolution);
    }
    Serial.println("stopeed");
  }

  delay(500);  // Wait for half a second before the next reading
}


