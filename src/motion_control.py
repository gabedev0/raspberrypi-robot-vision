import RPi.GPIO as GPIO

IN1, IN2, ENA = 17, 27, 18
IN3, IN4, ENB = 22, 23, 24

PWM_FREQ = 1000
DEFAULT_SPEED = 70

def gpio_setup():
    GPIO.setmode(GPIO.BCM)
    for p in (IN1, IN2, IN3, IN4, ENA, ENB):
        GPIO.setup(p, GPIO.OUT)
    pwmA = GPIO.PWM(ENA, PWM_FREQ)
    pwmB = GPIO.PWM(ENB, PWM_FREQ)
    pwmA.start(0)
    pwmB.start(0)
    return pwmA, pwmB

def stop_motors(pwmA, pwmB):
    GPIO.output(IN1, False); GPIO.output(IN2, False)
    GPIO.output(IN3, False); GPIO.output(IN4, False)
    pwmA.ChangeDutyCycle(0); pwmB.ChangeDutyCycle(0)

def forward(pwmA, pwmB, speed=DEFAULT_SPEED):
    GPIO.output(IN1, True); GPIO.output(IN2, False)
    GPIO.output(IN3, True); GPIO.output(IN4, False)
    pwmA.ChangeDutyCycle(speed); pwmB.ChangeDutyCycle(speed)

def backward(pwmA, pwmB, speed=DEFAULT_SPEED):
    GPIO.output(IN1, False); GPIO.output(IN2, True)
    GPIO.output(IN3, False); GPIO.output(IN4, True)
    pwmA.ChangeDutyCycle(speed); pwmB.ChangeDutyCycle(speed)

def turn_left(pwmA, pwmB, speed=DEFAULT_SPEED):
    GPIO.output(IN1, False); GPIO.output(IN2, True)
    GPIO.output(IN3, True); GPIO.output(IN4, False)
    pwmA.ChangeDutyCycle(speed); pwmB.ChangeDutyCycle(speed)

def turn_right(pwmA, pwmB, speed=DEFAULT_SPEED):
    GPIO.output(IN1, True); GPIO.output(IN2, False)
    GPIO.output(IN3, False); GPIO.output(IN4, True)
    pwmA.ChangeDutyCycle(speed); pwmB.ChangeDutyCycle(speed)
