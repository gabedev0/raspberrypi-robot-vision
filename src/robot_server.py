import socket
import logging
from motion_control import MotorController
import threading

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("robot_server")

HOST = "" 
PORT = 8000

def client_thread(conn, addr, motor):
    log.info("Conexão recebida de %s", addr)
    try:
        with conn:
            buf = b""
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                buf += data
                # suporta comandos terminados em newline
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    cmd = line.decode().strip().upper()
                    log.info("Comando: %s", cmd)
                    if cmd == "FORWARD":
                        motor.forward()
                    elif cmd == "BACK":
                        motor.backward()
                    elif cmd == "LEFT":
                        motor.left()
                    elif cmd == "RIGHT":
                        motor.right()
                    elif cmd == "STOP":
                        motor.stop()
                    elif cmd == "CLEANUP":
                        motor.cleanup()
                    else:
                        log.warning("Comando desconhecido: %s", cmd)
    except Exception as e:
        log.exception("Erro na thread do cliente: %s", e)

def run_server(host=HOST, port=PORT):
    motor = MotorController()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    log.info("Server listening on %s:%d", host, port)
    try:
        while True:
            conn, addr = s.accept()
            threading.Thread(target=client_thread, args=(conn, addr, motor), daemon=True).start()
    finally:
        motor.cleanup()
        s.close()

if __name__ == "__main__":
    run_server()
