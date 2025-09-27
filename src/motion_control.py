import RPi.GPIO as GPIO
import logging
import time

log = logging.getLogger("motion_control")
log.setLevel(logging.INFO)

IN1, IN2, ENA = 17, 27, 18
IN3, IN4, ENB = 22, 23, 24

PWM_FREQ = 1000
DEFAULT_SPEED = 60

class MotorController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        for p in (IN1, IN2, IN3, IN4, ENA, ENB):
            GPIO.setup(p, GPIO.OUT)
        self.pwmA = GPIO.PWM(ENA, PWM_FREQ)
        self.pwmB = GPIO.PWM(ENB, PWM_FREQ)
        self.pwmA.start(0)
        self.pwmB.start(0)

    def stop(self):
        self.pwmA.ChangeDutyCycle(0)
        self.pwmB.ChangeDutyCycle(0)
        log.info("Motors stopped.")

    def forward(self, speed=DEFAULT_SPEED):
        GPIO.output(IN1, True); GPIO.output(IN2, False)
        GPIO.output(IN3, True); GPIO.output(IN4, False)
        self.pwmA.ChangeDutyCycle(speed); self.pwmB.ChangeDutyCycle(speed)
        log.info("Forward at %d%%", speed)

    def backward(self, speed=DEFAULT_SPEED):
        GPIO.output(IN1, False); GPIO.output(IN2, True)
        GPIO.output(IN3, False); GPIO.output(IN4, True)
        self.pwmA.ChangeDutyCycle(speed); self.pwmB.ChangeDutyCycle(speed)
        log.info("Backward at %d%%", speed)

    def left(self, speed=DEFAULT_SPEED):
        GPIO.output(IN1, False); GPIO.output(IN2, True)
        GPIO.output(IN3, True); GPIO.output(IN4, False)
        self.pwmA.ChangeDutyCycle(speed); self.pwmB.ChangeDutyCycle(speed)
        log.info("Left at %d%%", speed)

    def right(self, speed=DEFAULT_SPEED):
        GPIO.output(IN1, True); GPIO.output(IN2, False)
        GPIO.output(IN3, False); GPIO.output(IN4, True)
        self.pwmA.ChangeDutyCycle(speed); self.pwmB.ChangeDutyCycle(speed)
        log.info("Right at %d%%", speed)

    def cleanup(self):
        self.stop()
        GPIO.cleanup()
        log.info("GPIO cleaned up.")
