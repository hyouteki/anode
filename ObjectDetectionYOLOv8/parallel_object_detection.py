from ultralytics import YOLO
from math import ceil
from typer import Typer
from queue import Queue
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from cv2 import (
    VideoCapture,
    imshow,
    waitKey,
    VideoWriter,
    putText,
    getTextSize,
    destroyAllWindows,
    rectangle,
    VideoWriter_fourcc,
    FONT_HERSHEY_SIMPLEX,
    FILLED,
)

COLOR_RECTANGLE = (223, 65, 80)[::-1]
COLOR_TEXT = (255, 255, 255)[::-1]
COLOR_OUTLINE = (32, 190, 175)[::-1]

FRAMES_AT_A_TIME = 6
MODEL_PATH = "model/yolov8l.pt"
COCO_PATH = "model/coco.names"

app = Typer()

model = YOLO(MODEL_PATH)
with open(COCO_PATH) as file:
    classes = file.read().rstrip("\n").split("\n")


def make_bounding_rectangle(frame, bounding_box, thickness=2, color=COLOR_RECTANGLE):
    if thickness != 0:
        rectangle(frame, bounding_box, color=color, thickness=thickness)
    return frame


def make_text_box(
    frame,
    text,
    pos,
    scale=0.7,
    thickness=2,
    color_text=COLOR_TEXT,
    color_rectangle=COLOR_RECTANGLE,
    font=FONT_HERSHEY_SIMPLEX,
    offset=0,
    border=None,
    color_outline=COLOR_OUTLINE,
):
    ox, oy = pos
    (w, h), _ = getTextSize(text, font, scale, thickness)
    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    rectangle(frame, (x1, y1), (x2 + 10, y2 - 10), color_rectangle, FILLED)
    if border is not None:
        rectangle(frame, (x1, y1), (x2, y2), color_outline, border)
    putText(frame, text, (ox + 5, oy - 5), font, scale, color_text, thickness)

    return frame, [x1, y2, x2, y1]


def process_frame(frame, frame_id, local_model):
    results = local_model(frame, stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            make_bounding_rectangle(frame=frame, bounding_box=(x1, y1, w, h))
            confidence = ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            make_text_box(
                frame, f"{classes[cls]} {confidence}", (max(0, x1), max(35, y1))
            )
    print(f"frame_{frame_id}")
    return frame


def make_annotation(video_capture, output_path: str = None):
    out = None
    if output_path is not None:
        fourcc = VideoWriter_fourcc(*"mp4v")
        frame_width = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))
        fps = int(video_capture.get(5))
        out = VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    executor = ThreadPoolExecutor(max_workers=FRAMES_AT_A_TIME)
    local_models = [YOLO(MODEL_PATH) for _ in range(6)]
    frame_queue = Queue()
    futures = []
    frame_id = 0
    expected_frame_id = 1

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame_id += 1

        if len(futures) < FRAMES_AT_A_TIME:
            local_model = local_models[len(futures)]
            futures.append(
                executor.submit(process_frame, frame.copy(), frame_id, local_model)
            )

        elif len(futures) == FRAMES_AT_A_TIME:
            for completed_future in futures:
                processed_frame = completed_future.result()
                frame_queue.put((frame_id, processed_frame))
                waitKey(1)

            futures.clear()

            while not frame_queue.empty():
                frame_number, processed_frame = frame_queue.get()
                if frame_number == expected_frame_id:
                    imshow("Image", processed_frame)
                    if output_path is not None:
                        out.write(processed_frame)
                    waitKey(1)
                    expected_frame_id += 1
                else:
                    frame_queue.put((frame_number, processed_frame))

    video_capture.release()
    for local_model in local_models:
        local_model.close()
    if output_path is not None:
        out.release()
    destroyAllWindows()


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
