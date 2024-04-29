import torch
from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
)
from argparse import ArgumentParser
from ultralytics import YOLO
from json import load
from keyboard import is_pressed
from threading import Thread
from math import ceil
import tensorflow as tf
from cv2 import (
    VideoCapture,
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

# Choose the `slowfast_r50` model
model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)

# Set to GPU or CPU
device = "cpu"
model = model.eval()
model = model.to(device)

json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try:
    urllib.URLopener().retrieve(json_url, json_filename)
except:
    urllib.request.urlretrieve(json_url, json_filename)

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10
num_crops = 3


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size),
            PackPathway(),
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate) / frames_per_second

# url_link = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
# video_path = "archery.mp4"
# try:
#     urllib.URLopener().retrieve(url_link, video_path)
# except:
#     urllib.request.urlretrieve(url_link, video_path)

# Select the duration of the clip to load by specifying the start and end duration
# The start_sec should correspond to where the action occurs in the video
start_sec = 0
end_sec = start_sec + clip_duration

# Initialize an EncodedVideo helper class and load the video
video = EncodedVideo.from_path(video_path)

# Load the desired clip
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

# Apply a transform to normalize the video input
video_data = transform(video_data)

# Move the inputs to the desired device
inputs = video_data["video"]
inputs = [i.to(device)[None, ...] for i in inputs]


# Pass the input clip through the model
preds = model(inputs)

# Get the predicted classes
post_act = torch.nn.Softmax(dim=1)
preds = post_act(preds)
pred_classes = preds.topk(k=5).indices[0]

print(pred_classes)

# Map the predicted classes to the label names
pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
print(
    "Top 5 predicted labels: %s" % ", ".join(pred_class_names)
)  # Pass the input clip through the model

COLOR_RECTANGLE = (223, 65, 80)[::-1]
COLOR_TEXT = (255, 255, 255)[::-1]


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


videostream = VideoStream().start()

while True:
    frame = videostream.read()
    video = EncodedVideo.from_path(video_path)
    results = MODEL(cvtColor(frame, COLOR_BGR2RGB), stream=True)

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

    imshow("pod yolov8", frame)

    if is_pressed("q"):
        break
    waitKey(1)

destroyAllWindows()
videostream.stop()
