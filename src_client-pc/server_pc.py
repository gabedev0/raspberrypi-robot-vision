import socket
import struct
import threading
import logging
import time
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("server_pc")

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

def map_classes_to_command(detected_classes):
    if not detected_classes:
        return "STOP"
    if 0 in detected_classes:
        return "FORWARD"
    if 2 in detected_classes:
        return "LEFT"
    return "STOP"

class ClientHandler(threading.Thread):
    def __init__(self, conn, addr, model, imgsz=640):
        super().__init__(daemon=True)
        self.conn = conn
        self.addr = addr
        self.model = model
        self.imgsz = imgsz
        self.running = True
        self.prev_time = time.perf_counter()
        self.fps = 0.0
        self.alpha = 0.9

    def run(self):
        log.info("Cliente conectado: %s", self.addr)
        try:
            while self.running:
                hdr = recvall(self.conn, 4)
                if not hdr:
                    log.info("Conexão fechada pelo cliente %s", self.addr)
                    break
                size = struct.unpack(">I", hdr)[0]
                if size <= 0 or size > 20_000_000:
                    log.warning("Tamanho inválido recebido: %d", size)
                    break
                jpg = recvall(self.conn, size)
                if not jpg:
                    log.info("Erro recebendo dados JPG")
                    break

                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    log.warning("Falha ao decodificar JPEG")
                    continue

                now = time.perf_counter()
                dt = now - self.prev_time
                self.prev_time = now
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self.fps = inst_fps if self.fps == 0.0 else self.alpha * self.fps + (1 - self.alpha) * inst_fps

                results = self.model(frame, imgsz=self.imgsz)
                r = results[0]
                try:
                    annotated = r.plot()
                except Exception:
                    annotated = frame.copy()

                classes = []
                if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                    try:
                        cls = r.boxes.cls
                        classes = [int(x) for x in cls.tolist()]
                    except Exception:
                        for box in r.boxes:
                            try:
                                classes.append(int(box.cls))
                            except Exception:
                                pass

                cmd = map_classes_to_command(classes)

                try:
                    self.conn.sendall((cmd + "\n").encode("utf-8"))
                except Exception as e:
                    log.exception("Erro enviando comando: %s", e)
                    break

                cv2.putText(annotated, f"CMD: {cmd}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

                cv2.imshow(f"Client {self.addr}", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    log.info("Quit requested from display")
                    self.running = False
                    break

        except Exception as e:
            log.exception("Erro no handler do cliente: %s", e)
        finally:
            try:
                self.conn.close()
            except:
                pass
            log.info("Handler finalizado para %s", self.addr)

def run_server(host="0.0.0.0", port=8000, imgsz=640, model_name="yolov8n.pt", device="cpu"):
    log.info("Carregando modelo YOLO: %s (device=%s)...", model_name, device)
    model = YOLO(model_name)
    if device != "cpu":
        try:
            model.to(device)
        except Exception:
            log.warning("Falha ao mover modelo para device %s. Usando CPU.", device)
            device = "cpu"

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    log.info("Servidor escutando em %s:%d", host, port)

    try:
        while True:
            conn, addr = s.accept()
            handler = ClientHandler(conn, addr, model, imgsz=imgsz)
            handler.start()
    except KeyboardInterrupt:
        log.info("Servidor encerrando por KeyboardInterrupt")
    finally:
        s.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--imgsz", type=int, default=640, help="tamanho imgsz para inferência (yolov8 imgsz)")
    p.add_argument("--model", default="yolov8n.pt", help="caminho ou nome do modelo YOLO")
    p.add_argument("--device", default="cpu", help="device para rodar (cpu ou cuda)")
    args = p.parse_args()
    run_server(args.host, args.port, imgsz=args.imgsz, model_name=args.model, device=args.device)
