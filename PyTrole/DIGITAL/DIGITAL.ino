#include <Servo.h>

#define TRIGGER_PIN 8
#define ECHO_PIN 9
#define SERVO_PIN 11
#define T 0.005
Servo servo;

float getInput(){
    digitalWrite(TRIGGER_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIGGER_PIN, LOW);
    delayMicroseconds(10);

    float duration = pulseIn(ECHO_PIN, HIGH);
    return duration * 0.034 / 2;
}

float applyStep(int magnitude){
    int output = mapFloat(magnitude, -100, 100, 45, 135);
    servo.write(output);
}

double computeController(double inputValue, double setpoint){
    static double Y_n1, Y_n2;
    static double X_n1, X_n2;
    
    double X_n = setpoint - inputValue;
    double Y_n = -7.268431752660472*X_n + 15.040765287341745*X_n1 -7.771048331979746*X_n2 + 1.951219512195122*Y_n1-0.951219512195122*Y_n2;

    if(Y_n < -100)
        Y_n = -100;

    if(Y_n > 100)
        Y_n = 100;
    
    X_n2 = X_n1;
    X_n1 = X_n; 
    Y_n2 = Y_n1;
    Y_n1 = Y_n;
    return Y_n;
}

void setup() {
    pinMode(TRIGGER_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    servo.attach(SERVO_PIN);
    
    Serial.begin(9600);
    applyStep(0);

    Serial.println("INICIANDO...");
    delay(2000);
}

void loop() {
    static unsigned long time = millis();
    double time_passed = (millis() - time)*1e-3;

    if(time_passed < T)
        return;
    
    time = millis();

    float setpoint = 10;
    float input  = getInput();

    float output = computeController(input, setpoint);
    applyStep(output);
    
    //Serial.println("input:  " + String(input));
    //Serial.println("output: " + String(output));
    //Serial.println("erro:   " + String(setpoint-input));
    Serial.println();
}
