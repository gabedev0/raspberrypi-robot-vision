import socket
import struct
import time
import argparse
import logging
import sys

import cv2
import numpy as np

USE_PICAMERA2 = False
try:
    from picamera2 import Picamera2
    USE_PICAMERA2 = True
except Exception:
    USE_PICAMERA2 = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("robot_client")

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

class Camera:
    def __init__(self, width=640, height=360, use_picamera=True, src=0):
        self.width = width
        self.height = height
        self.use_picamera = use_picamera and USE_PICAMERA2
        self.src = src
        if self.use_picamera:
            log.info("Iniciando Picamera2 (libcamera)")
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration({"size": (self.width, self.height)})
            config['main']['format'] = 'RGB888'
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(0.2)
        else:
            log.info("Iniciando VideoCapture (OpenCV V4L2) src=%s", src)
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                raise RuntimeError("Falha ao abrir VideoCapture. Verifique camera/libcamera/legacy driver.")

    def read(self):
        if self.use_picamera:
            arr = self.picam2.capture_array()
            if arr is None:
                return None
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return frame
        else:
            ret, frame = self.cap.read()
            return frame if ret else None

    def release(self):
        if self.use_picamera:
            try:
                self.picam2.stop()
            except Exception:
                pass
        else:
            try:
                self.cap.release()
            except Exception:
                pass

def run_stream(server_ip, server_port=8000, width=640, height=360, quality=80, use_picamera=True):
    cam = Camera(width=width, height=height, use_picamera=use_picamera)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    log.info("Conectando ao servidor %s:%d ...", server_ip, server_port)
    sock.connect((server_ip, server_port))
    sock.settimeout(3.0)  
    log.info("Conectado. Iniciando streaming de frames... (pressione CTRL+C para sair)")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                log.debug("Frame None - pulando")
                time.sleep(0.01)
                continue

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            ret, buf = cv2.imencode(".jpg", frame, encode_param)
            if not ret:
                log.warning("Falha ao codificar frame JPEG")
                continue
            data = buf.tobytes()
            size = len(data)
            sock.sendall(struct.pack(">I", size) + data)

            cmd = b""
            start = time.time()
            try:
                while True:
                    ch = sock.recv(1)
                    if not ch:
                        raise ConnectionError("Conexão fechada pelo servidor")
                    if ch == b"\n":
                        break
                    cmd += ch
                    if time.time() - start > 2.5:  # 2.5s timeout para comando
                        break
            except socket.timeout:
                log.debug("Timeout aguardando comando")
                cmd = b""
            except Exception as e:
                log.exception("Erro lendo comando: %s", e)
                raise

            cmd_str = cmd.decode('utf-8', errors='ignore').strip()
            if cmd_str:
                log.info("Comando recebido: %s", cmd_str)
            else:
                log.debug("Nenhum comando recebido (vazio)")

            time.sleep(0.01)
    except KeyboardInterrupt:
        log.info("Encerrando por KeyboardInterrupt")
    finally:
        try:
            sock.close()
        except:
            pass
        cam.release()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="IP do PC servidor (ex: 192.168.1.50)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    p.add_argument("--no-picam", dest="use_picamera", action="store_false", help="Forçar uso OpenCV/V4L2")
    args = p.parse_args()
    run_stream(args.server, args.port, args.width, args.height, args.quality, use_picamera=args.use_picamera)
