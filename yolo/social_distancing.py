import cv2
import numpy as np
from ultralytics import YOLO
from json import load
from argparse import ArgumentParser
from math import ceil
from cv2 import (
    cvtColor,
    COLOR_BGR2RGB,
    putText,
    FONT_HERSHEY_SIMPLEX,
    imshow,
    waitKey,
    destroyAllWindows,
)

def calculate_distance(box1, box2):
    # Calculate Euclidean distance between the centroids of two bounding boxes
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    centroid1 = ((x1 + x2) // 2, (y1 + y2) // 2)
    centroid2 = ((x3 + x4) // 2, (y3 + y4) // 2)
    return np.linalg.norm(np.array(centroid1) - np.array(centroid2))

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

# Initialize video capture
cap = cv2.VideoCapture("assets/town.avi")

# Initialize violation count
violation_count = 0
video_width = int(cap.get(3))
video_height = int(cap.get(4))
cv2.namedWindow("Social Distancing Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and preprocess frame
    outs = MODEL(cvtColor(frame, COLOR_BGR2RGB), stream=True)
    people_boxes = []

    # Reset violation count for each frame
    frame_violation_count = 0

    # Process detection results
    for out in outs:
        for box in out.boxes:

            confidence = ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = f"{CLASSES[cls]} {int(confidence*100)}%"

            if confidence > 0.5 and CLASSES[cls] == "person":
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                people_boxes.append((x1, y1, x2, y2))

    # Check for social distancing violations
    for i in range(len(people_boxes)):
        for j in range(i+1, len(people_boxes)):
            # Calculate distance between people
            dist = calculate_distance(people_boxes[i], people_boxes[j])
            if dist < 70:  # Adjust this distance threshold as needed for your scenario
                # Draw bounding boxes around people in violation
                x1, y1, x2, y2 = people_boxes[i]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                x1, y1, x2, y2 = people_boxes[j]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                frame_violation_count += 1
                

    # Update the total violation count
    violation_count += frame_violation_count

    # Draw bounding boxes, lines, and violation count on the frame
    cv2.putText(frame, f"Violations: {violation_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    window_aspect_ratio = video_width / video_height
    window_height = 720  # You can adjust this height as needed
    window_width = int(window_height * window_aspect_ratio)
    frame = cv2.resize(frame, (window_width, window_height))

    # Display frame
    cv2.imshow("Social Distancing Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
