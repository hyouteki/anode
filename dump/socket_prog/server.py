from socket import socket, AF_INET, SOCK_STREAM, gethostname
from pickle import loads
from struct import calcsize, unpack
from argparse import ArgumentParser
from ultralytics import YOLO
from json import load
from keyboard import is_pressed
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

COLOR_RECTANGLE = (223, 65, 80)[::-1]
COLOR_TEXT = (255, 255, 255)[::-1]


def object_detection(frame, model):
    results = model(cvtColor(frame, COLOR_BGR2RGB), stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            rectangle(frame, (x1, y1), (x2, y2), color=(223, 65, 80), thickness=2)
            confidence = int(box.conf[0] * 100)
            cls = int(box.cls[0])
            ox, oy = max(0, x1), max(35, y1)
            label = f"{CLASSES[cls]} {int(confidence)}%"
            (w, h), _ = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.7, 1)
            x1, y1, x2, y2 = ox, oy, ox + w, oy - h
            rectangle(frame, (x1, y1), (x2 + 10, y2 - 10), COLOR_RECTANGLE, FILLED)
            putText(
                frame, label, (ox + 5, oy - 5), FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1
            )


if __name__ == "__main__":

    def get_config_file():
        parser = ArgumentParser()
        parser.add_argument(
            "--config",
            help="where the .config file is located",
            default=".config",
        )

        args = parser.parse_args()
        CONFIG_PATH = args.config
        with open(CONFIG_PATH, "r") as file:
            return load(file)

    CONFIG = get_config_file()

    MODEL = YOLO(CONFIG["object detection"]["model"])
    with open(CONFIG["object detection"]["coco"], "r") as file:
        CLASSES = file.read().rstrip("\n").split("\n")

    server_socket = socket(AF_INET, SOCK_STREAM)
    server_socket.bind((gethostname(), CONFIG["socket"]["port"]))
    server_socket.listen(CONFIG["socket"]["max clients"])
    print("Listening for incoming connections...")

    client_socket, addr = server_socket.accept()
    print("Received connection from", addr)

    data = b""
    payload_size = calcsize("Q")
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4 * 1024)
            if not packet:
                break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4 * 1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = loads(frame_data)
        object_detection(frame, MODEL)
        imshow("Server", frame)

        if is_pressed("q"):
            break
        waitKey(1)

    client_socket.close()
    server_socket.close()
    destroyAllWindows()
