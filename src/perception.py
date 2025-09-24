import time
import threading
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

class CameraThread:
    def __init__(self, src=0, width=640, height=360, queue_size=2):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível abrir a câmera (verifique libcamera / raspicam).")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q = deque(maxlen=queue_size)
        self.stopped = False
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self.q.append(frame)

    def read(self):
        if self.q:
            return self.q.pop()
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

    def release(self):
        self.stopped = True
        self.t.join(timeout=0.5)
        self.cap.release()

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt", img_size=320, conf=0.25, iou=0.45, max_det=10):
        self.model = YOLO(model_name)
        self.img_size = img_size
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.names_map = self.model.names if hasattr(self.model, "names") else {}

    def detect(self, frame, wanted_idxs, scale=None):
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model.predict(
            source=img_rgb,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            max_det=self.max_det,
            device="cpu",
            verbose=False
        )
        r = results[0]
        boxes, scores, classes = [], [], []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls = int(b.cls.cpu().numpy().ravel())
                if cls not in wanted_idxs:
                    continue
                xyxy = b.xyxy.cpu().numpy().ravel()
                conf = float(b.conf.cpu().numpy().ravel())
                if scale and scale < 1.0:
                    xyxy = xyxy / scale
                boxes.append(xyxy)
                scores.append(conf)
                classes.append(cls)
        return boxes, scores, classes

def draw_boxes(frame, boxes, scores, classes, names):
    for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - 18), (x1 + t[0] + 6, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame
