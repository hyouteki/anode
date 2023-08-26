from ultralytics import YOLO
from cv2 import VideoCapture, imshow, waitKey
from cvzone import cornerRect, putTextRect
from math import ceil
from time import time
from typer import Typer

app = Typer()

model = YOLO("model/yolov8l.pt")
with open("model/coco.names", "r") as file:
    classes = file.read().rstrip('\n').split('\n')

def make_annotation(video_capture):
    prev_frame_time = 0
    new_frame_time = 0
    while True:
        new_frame_time = time()
        success, img = video_capture.read()
        results = model(img, stream=True)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                cornerRect(img, (x1, y1, w, h))
                confidence = ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                putTextRect(img, f'{classes[cls]} {confidence}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)

        imshow("Image", img)
        waitKey(1)

@app.command()
def webcam():
    video_capture = VideoCapture(1)
    video_capture.set(3, 1280)
    video_capture.set(4, 720)
    make_annotation(video_capture)

@app.command()
def video(path: str):
    video_capture = VideoCapture(path)
    make_annotation(video_capture)

if __name__ == "__main__":
    app()