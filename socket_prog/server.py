import socket
import cv2
import pickle
import struct
from threading import Thread
from argparse import ArgumentParser
from ultralytics import YOLO
from json import load
from cv2 import (
    cvtColor,
    COLOR_BGR2RGB,
    rectangle,
    getTextSize,
    FONT_HERSHEY_SIMPLEX,
    FILLED,
    putText,
    imshow,
    waitKey,
    destroyAllWindows,
)

def object_detection(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            rectangle(frame, (x1, y1), (x2, y2), color=(223, 65, 80), thickness=2)
            confidence = int(box.conf[0] * 100)
            cls = int(box.cls[0])
            label = f"{CLASSES[cls]} {confidence}%"
            (w, h), _ = cv2.getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.7, 1)
            x1, y1, x2, y2 = x1, y1 - 20, x1 + w, y1
            rectangle(frame, (x1, y1), (x2 + 10, y2 - 10), (223, 65, 80), FILLED)
            putText(frame, label, (x1 + 5, y1 - 5), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return frame

def open_camera_window(window_name, frame):
    cv2.imshow(window_name, frame)

parser = ArgumentParser()
parser.add_argument(
    "--params",
    help="where the parameters (JSON) file is located",
    default="params.json",
)

args = parser.parse_args()
PARAMS_PATH = args.params

with open(PARAMS_PATH, "r") as file:
    params = load(file)

MODEL = YOLO(params["model"])
with open(params["coco"], "r") as file:
    CLASSES = file.read().rstrip("\n").split("\n")

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.43.95'  # Replace with your server's IP address
port = 9999
server_socket.bind((host_ip, port))
server_socket.listen(5)
print("Listening for incoming connections...")

while True:
    client_socket, addr = server_socket.accept()
    print('Received connection from', addr)

    data = b""
    payload_size = struct.calcsize("Q")
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4 * 1024)
            if not packet:
                break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4 * 1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        # Perform object detection and annotate the frame
        annotated_frame = object_detection(frame, MODEL)

        # Display the annotated frame locally
        open_camera_window("Server Camera", annotated_frame)

        # Send the annotated frame back to the client
        annotated_data = pickle.dumps(annotated_frame)
        client_socket.sendall(struct.pack("Q", len(annotated_data)) + annotated_data)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    client_socket.close()

server_socket.close()
destroyAllWindows()
