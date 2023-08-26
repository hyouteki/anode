from ultralytics import YOLO
from cv2 import VideoCapture, imshow, waitKey, VideoWriter, VideoWriter_fourcc
from cvzone import cornerRect, putTextRect
from math import ceil
from time import time
from typer import Typer

app = Typer()

model = YOLO("model/yolov8l.pt")
with open("model/coco.names", "r") as file:
    classes = file.read().rstrip('\n').split('\n')

def make_annotation(video_capture, output_path: str = None):
    prev_frame_time = 0
    new_frame_time = 0
    frame_count = 1

    out = None
    if output_path != None:
        fourcc = VideoWriter_fourcc(*'mp4v')
        frame_width = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))
        fps = int(video_capture.get(5))
        out = VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        new_frame_time = time()
        success, frame = video_capture.read()

        if not success:
            break
        results = model(frame, stream=True)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                cornerRect(frame, (x1, y1, w, h))
                confidence = ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                putTextRect(frame, f'{classes[cls]} {confidence}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(f"frame_{frame_count}: {fps}")
        frame_count += 1

        imshow("Image", frame)
        if output_path != None:
            out.write(frame)
        waitKey(1)

    if output_path != None:
        out.release()
    video_capture.release()

@app.command()
def webcam():
    video_capture = VideoCapture(1)
    video_capture.set(3, 1280)
    video_capture.set(4, 720)
    make_annotation(video_capture)

@app.command()
def video(path: str, output_path: str = None):
    video_capture = VideoCapture(path)
    make_annotation(video_capture, output_path)

if __name__ == "__main__":
    app()