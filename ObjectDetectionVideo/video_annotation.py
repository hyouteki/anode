"""
@requirements
opencv-python
openh264-1.8.0-win64.dll
ffmpeg
yolov3.weights
yolov3.cfg
coco.names
"""

from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc, rectangle 
from cv2 import putText, FONT_HERSHEY_SIMPLEX, destroyAllWindows, resize
from cv2.dnn import readNet, blobFromImage

# Load YOLOv3 model
net = readNet(r"model/yolov3.weights", "model/yolov3.cfg")
with open("model/coco.names", "r") as file:
    classes = file.read().rstrip('\n').split('\n')

# Input and output video paths
input_video_path = 'assets/puss.mp4'
output_video_path = 'output_annotated_cat2.mp4'

# Open the input video
video_capture = VideoCapture(input_video_path)

# Get video properties
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fps = int(video_capture.get(5))

# Create VideoWriter object to save the output video
fourcc = VideoWriter_fourcc(*'mp4v')
out = VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break

    # resizing frame to match YOLO requirements
    aspect_ratio = frame.shape[1] / frame.shape[0]  # width / height
    # new_width = 416
    # new_height = int(new_width / aspect_ratio)
    # frame = resize(frame, (416, 416))
    # print(frame.shape)

    # Preprocess input frame for YOLO
    blob = blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # blob = blob[0, :, :, :]
    # print(blob.shape)

    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process detection results and draw rectangles
    for tmp in outs:
        for detection in tmp:
            # print(detection)
            scores = detection[5:]
            class_id = int(detection[1])
            confidence = scores[class_id]
            
            if confidence > 0.1:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw rectangle and label on the frame
                rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                putText(frame, classes[class_id], (x, y - 10), FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the annotated frame to the output video
    out.write(frame)

# Release video capture and writer objects
video_capture.release()
out.release()
destroyAllWindows()