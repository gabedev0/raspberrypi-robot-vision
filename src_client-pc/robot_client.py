import cv2
import socket
import struct
import pickle
import RPi.GPIO as GPIO

# Controle do robô
class RobotControl:
    def __init__(self, in1=17, in2=18, in3=22, in4=23, ena=24, enb=25):
        self.in1, self.in2, self.in3, self.in4 = in1, in2, in3, in4
        self.ena, self.enb = ena, enb

        GPIO.setmode(GPIO.BCM)
        GPIO.setup([in1, in2, in3, in4, ena, enb], GPIO.OUT)

        self.pwm_a = GPIO.PWM(ena, 1000)
        self.pwm_b = GPIO.PWM(enb, 1000)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

    def set_speed(self, left_speed, right_speed):
        self.pwm_a.ChangeDutyCycle(abs(left_speed))
        self.pwm_b.ChangeDutyCycle(abs(right_speed))

        GPIO.output(self.in1, left_speed > 0)
        GPIO.output(self.in2, left_speed < 0)
        GPIO.output(self.in3, right_speed > 0)
        GPIO.output(self.in4, right_speed < 0)

    def stop(self):
        self.set_speed(0, 0)

    def cleanup(self):
        self.stop()
        GPIO.cleanup()

# Cliente socket
SERVER_IP = "0.0.0.0"  #IP do PC
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, PORT))

cap = cv2.VideoCapture(0)
robot = RobotControl()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Serializa o frame
        data = pickle.dumps(frame)
        msg = struct.pack("Q", len(data)) + data
        sock.sendall(msg)

        # Recebe comando do servidor
        command = sock.recv(1024).decode("utf-8").strip()

        if command == "FORWARD":
            robot.set_speed(50, 50)
        elif command == "BACK":
            robot.set_speed(-50, -50)
        elif command == "LEFT":
            robot.set_speed(-40, 40)
        elif command == "RIGHT":
            robot.set_speed(40, -40)
        else:
            robot.stop()

finally:
    cap.release()
    robot.cleanup()
    sock.close()
