import socket
import time
import argparse
import cv2
import logging

from perception import CameraThread, ObjectDetector

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

CAM_INDEX = 0
CAP_WIDTH, CAP_HEIGHT = 640, 360
IMG_SZ = 320
MODEL_NAME = "yolov8n.pt"
CONF_THRESH, IOU_THRESH = 0.25, 0.45

def map_classes_to_command(detected_classes):
    """
    Mapeamento de exemplo — ajuste para os classes IDs do teu modelo/dataset.
    """
    if not detected_classes:
        return "STOP"
    if 0 in detected_classes:
        return "FORWARD"
    if 2 in detected_classes:
        return "LEFT"
    return "STOP"

def run_client(server_ip, server_port=8000, use_picamera=True):
    cam = CameraThread(src=CAM_INDEX, width=CAP_WIDTH, height=CAP_HEIGHT, use_picamera=use_picamera)
    cam.start()
    detector = ObjectDetector(model_path=MODEL_NAME, conf=CONF_THRESH, iou=IOU_THRESH, device="cpu")

    # Conecta ao servidor de controle (robot)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    log.info("Conectando em %s:%d ...", server_ip, server_port)
    sock.connect((server_ip, server_port))
    sock.settimeout(2.0)

    # FPS vars
    prev_time = time.perf_counter()
    fps = 0.0
    alpha = 0.9 

    try:
        while True:
            frame = cam.read(timeout=2.0)
            if frame is None:
                log.warning("Nenhum frame recebido (timeout).")
                time.sleep(0.1)
                continue

            # calcular FPS
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                if fps == 0.0:
                    fps = inst_fps
                else:
                    fps = alpha * fps + (1.0 - alpha) * inst_fps

            annotated, classes = detector.infer(frame, imgsz=IMG_SZ)
            cmd = map_classes_to_command(classes)
            try:
                sock.sendall((cmd + "\n").encode("utf-8"))
            except Exception as e:
                log.exception("Erro enviando comando: %s", e)
                break

            # desenhar texto de comando e FPS
            cv2.putText(annotated, f"CMD: {cmd}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("YOLOv8 - annotated", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        log.info("Encerrando cliente.")
        cam.stop()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="IP do servidor/robot que receberá comandos")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-picam", dest="use_picamera", action="store_false", help="Forçar uso OpenCV/V4L2 (legacy)")
    args = p.parse_args()
    run_client(args.server, args.port, use_picamera=args.use_picamera)
