# src/perception.py
import time
import threading
from collections import deque
import logging

import cv2
import numpy as np
from ultralytics import YOLO

PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception as e:
    PICAMERA2_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("perception")

class CameraThread(threading.Thread):
    def __init__(self, src=0, width=640, height=360, queue_size=2, use_picamera=True):
        super().__init__(daemon=True)
        self.width = width
        self.height = height
        self.queue = deque(maxlen=queue_size)
        self.running = False
        self.use_picamera = use_picamera and PICAMERA2_AVAILABLE
        self.src = src

        if self.use_picamera:
            log.info("Usando Picamera2 para captura.")
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration({'size': (self.width, self.height)})
            config['main']['format'] = 'RGB888'
            self.picam2.configure(config)
        else:
            log.info("Usando OpenCV VideoCapture para captura (fallback).")
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError("Não foi possível abrir VideoCapture (V4L2). Verifique libcamera/legacy driver.")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def run(self):
        self.running = True
        if self.use_picamera:
            self.picam2.start()
            while self.running:
                try:
                    arr = self.picam2.capture_array()
                    if arr is None:
                        log.warning("Picamera2 retornou None frame")
                        time.sleep(0.01)
                        continue
                    frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    self.queue.append(frame)
                except Exception as e:
                    log.exception("Erro lendo Picamera2: %s", e)
                    time.sleep(0.1)
        else:
            while self.running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    log.debug("VideoCapture read falhou; tentando novamente")
                    time.sleep(0.02)
                    continue
                self.queue.append(frame)

    def read(self, timeout=1.0):
        start = time.time()
        while time.time() - start < timeout:
            if len(self.queue) > 0:
                return self.queue[-1].copy()
            time.sleep(0.005)
        return None

    def stop(self):
        self.running = False
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

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", device="cpu", conf=0.25, iou=0.45):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou

    def infer(self, frame, imgsz=640):
        results = self.model(frame, imgsz=imgsz, conf=self.conf, iou=self.iou, device=self.device)
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
        return annotated, classes

def draw_boxes(img, results):
    return results
