import socket
import cv2
import pickle
import struct
from argparse import ArgumentParser
from ultralytics import YOLO
from json import load
from keyboard import is_pressed
from time import sleep
from threading import Thread
from math import ceil
from cv2 import (
    VideoCapture,
    getTickCount,
    getTickFrequency,
    cvtColor,
    LINE_AA,
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


class VideoStream:
    """Camera object that controls video streaming from the webcam"""

    def __init__(self):
        self.stream = VideoCapture(0)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


COLOR_RECTANGLE = (223, 65, 80)[::-1]
COLOR_TEXT = (255, 255, 255)[::-1]

parser = ArgumentParser()
parser.add_argument(
    "--params",
    help="where the parameters (JSON) file is located",
    default="params.json",
)
parser.add_argument(
    "--source",
    help="camera/webcam source",
    default=0,
)

args = parser.parse_args()
PARAMS_PATH = args.params
SOURCE = args.source

with open(PARAMS_PATH, "r") as file:
    params = load(file)
MODEL = YOLO(params["model"])
with open(params["coco"], "r") as file:
    CLASSES = file.read().rstrip("\n").split("\n")

frame_rate_calc = 1
freq = getTickFrequency()
videostream = VideoStream().start()
sleep(1)

# Socket Create
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('HOST IP:', host_ip)
port = 9999
socket_address = (host_ip, port)

# Socket Bind
server_socket.bind(socket_address)

# Socket Listen
server_socket.listen(5)
print("LISTENING AT:", socket_address)

# Socket Accept
while True:
    client_socket, addr = server_socket.accept()
    print('GOT CONNECTION FROM:', addr)
    if client_socket:
        while True:
            t1 = getTickCount()
            frame1 = videostream.read()
            frame = frame1.copy()
            frame_rgb = cvtColor(frame, COLOR_BGR2RGB)

            results = MODEL(frame_rgb, stream=True)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    rectangle(frame, (x1, y1), (x2, y2), color=COLOR_RECTANGLE, thickness=2)
                    confidence = ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    ox, oy = max(0, x1), max(35, y1)
                    label = f"{CLASSES[cls]} {int(confidence*100)}%"
                    (w, h), _ = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.7, 1)
                    x1, y1, x2, y2 = ox, oy, ox + w, oy - h
                    rectangle(frame, (x1, y1), (x2 + 10, y2 - 10), COLOR_RECTANGLE, FILLED)
                    putText(
                        frame, label, (ox + 5, oy - 5), FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1
                    )

            putText(
                frame,
                "FPS: {0:.2f}".format(frame_rate_calc),
                (30, 50),
                FONT_HERSHEY_SIMPLEX,
                1,
                (223, 65, 80)[::-1],
                2,
                LINE_AA,
            )

            imshow("Object detector", frame)
            t2 = getTickCount()
            time1 = (t2 - t1) / freq
            frame_rate_calc = 1 / time1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        client_socket.close()
        destroyAllWindows()
        videostream.stop()
        break
