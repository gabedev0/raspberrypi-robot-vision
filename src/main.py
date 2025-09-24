import time
import argparse
import cv2
import RPi.GPIO as GPIO

from perception import CameraThread, ObjectDetector, draw_boxes
from motion_control import gpio_setup, stop_motors, forward, turn_left, turn_right

# Configs
CAM_INDEX = 0
CAP_WIDTH, CAP_HEIGHT = 640, 360
IMG_SZ = 320
MODEL_NAME = "yolov8n.pt"
CONF_THRESH, IOU_THRESH = 0.25, 0.45
MAX_DETS = 10
DEFAULT_SPEED = 70

WANTED_CLASSES = ["person", "clock", "tvmonitor", "chair",
                "traffic light", "cell phone", "mouse", "keyboard"]
ALIAS = {"tv": "tvmonitor", "cellphone": "cell phone"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    args = parser.parse_args()
    show = not args.no_show

    # Detector
    detector = ObjectDetector(MODEL_NAME, IMG_SZ, args.conf, IOU_THRESH, MAX_DETS)
    names_map = detector.names_map
    wanted_idxs = {i for i, n in names_map.items() if n.lower() in [c.lower() for c in WANTED_CLASSES]}

    # Camera
    cam = CameraThread(src=CAM_INDEX, width=CAP_WIDTH, height=CAP_HEIGHT, queue_size=2)
    time.sleep(0.5)

    # GPIO
    pwmA, pwmB = gpio_setup()
    stop_motors(pwmA, pwmB)

    fps_smooth, alpha = 0.0, 0.2

    try:
        while True:
            t0 = time.time()
            frame = cam.read()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            scale = IMG_SZ / max(h, w)
            if scale < 1.0:
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                small = frame

            boxes, scores, classes = detector.detect(small, wanted_idxs, scale if scale < 1.0 else None)

            cmd_sent = "STOP"
            person_idxs = [i for i, c in enumerate(classes) if names_map[c].lower() == "person"]
            if person_idxs:
                best = max(person_idxs, key=lambda i: scores[i])
                x1, _, x2, _ = boxes[best]
                cx_ratio = (x1 + x2) / (2 * w)
                if cx_ratio < 0.35:
                    turn_left(pwmA, pwmB, speed=DEFAULT_SPEED)
                    cmd_sent = "LEFT"
                elif cx_ratio > 0.65:
                    turn_right(pwmA, pwmB, speed=DEFAULT_SPEED)
                    cmd_sent = "RIGHT"
                else:
                    forward(pwmA, pwmB, speed=DEFAULT_SPEED)
                    cmd_sent = "FORWARD"
            else:
                stop_motors(pwmA, pwmB)

            frame = draw_boxes(frame, boxes, scores, classes, names_map)
            cv2.putText(frame, f"CMD: {cmd_sent}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            t1 = time.time()
            fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0
            fps_smooth = (1 - alpha) * fps_smooth + alpha * fps
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if show:
                cv2.imshow("YOLO Robot", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                print(f"FPS: {fps_smooth:.1f} | Dets: {len(boxes)} | CMD: {cmd_sent}", end="\r")

    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")
    finally:
        cam.release()
        stop_motors(pwmA, pwmB)
        pwmA.stop(); pwmB.stop()
        GPIO.cleanup()
        if show:
            cv2.destroyAllWindows()
        print("\nEncerrado.")

if __name__ == "__main__":
    main()
