import socket
import struct
import pickle
import cv2
from ultralytics import YOLO

HOST = "0.0.0.0"
PORT = 5000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print("[INFO] Aguardando conexão do robô...")
client_socket, addr = server_socket.accept()
print(f"[INFO] Robô conectado: {addr}")

model = YOLO("yolov8n.pt")

data = b""
payload_size = struct.calcsize("Q")

try:
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet:
                break
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4*1024)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data)

        # Detecção
        results = model(frame, classes=[0,62,63,67,70,72,73], verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Servidor - Processamento", annotated)

        detected_classes = [int(c) for c in results[0].boxes.cls]

        # Decide comando
        if 0 in detected_classes:  # pessoa
            command = "FORWARD"
        elif 67 in detected_classes:  # semáforo
            command = "STOP"
        else:
            command = "STOP"

        client_socket.sendall(command.encode("utf-8"))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    client_socket.close()
    server_socket.close()
