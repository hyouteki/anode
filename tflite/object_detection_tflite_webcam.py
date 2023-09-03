from os import getcwd
from os.path import join
from argparse import ArgumentParser
from cv2 import (
    VideoCapture,
    getTickCount,
    getTickFrequency,
    cvtColor,
    resize,
    COLOR_BGR2RGB,
    rectangle,
    getTextSize,
    FONT_HERSHEY_SIMPLEX,
    FILLED,
    putText,
    LINE_AA,
    imshow,
    waitKey,
    destroyAllWindows,
)
from keyboard import is_pressed
from numpy import float32, expand_dims
from time import sleep
from threading import Thread
from importlib.util import find_spec


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


parser = ArgumentParser()
parser.add_argument(
    "--modeldir", help="Folder the .tflite file is located in", required=True
)
parser.add_argument(
    "--graph",
    help="Name of the .tflite file, if different than detect.tflite",
    default="detect.tflite",
)
parser.add_argument(
    "--labels",
    help="Name of the labelmap file, if different than labelmap.txt",
    default="labelmap.txt",
)
parser.add_argument(
    "--threshold",
    help="Minimum confidence threshold for displaying detected objects",
    default=0.5,
)
parser.add_argument(
    "--resolution",
    help="Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.",
    default="1280x720",
)
parser.add_argument(
    "--edgetpu",
    help="Use Coral Edge TPU Accelerator to speed up detection",
    action="store_true",
)

args = parser.parse_args()
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split("x")
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

if find_spec("tflite_runtime"):
    from tflite_runtime.interpreter import Interpreter

    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter

    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU and GRAPH_NAME == "detect.tflite":
    GRAPH_NAME = "edgetpu.tflite"

CWD_PATH = getcwd()
PATH_TO_CKPT = join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, "r") as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == "???":
    del labels[0]

if use_TPU:
    interpreter = Interpreter(
        model_path=PATH_TO_CKPT,
        experimental_delegates=[load_delegate("libedgetpu.so.1.0")],
    )
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]

floating_model = input_details[0]["dtype"] == float32

input_mean = 127.5
input_std = 127.5


outname = output_details[0]["name"]

if "StatefulPartitionedCall" in outname:  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_rate_calc = 1
freq = getTickFrequency()
videostream = VideoStream().start()
sleep(1)

while True:
    t1 = getTickCount()
    frame1 = videostream.read()
    frame = frame1.copy()
    frame_rgb = cvtColor(frame, COLOR_BGR2RGB)
    frame_resized = resize(frame_rgb, (width, height))
    input_data = expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]["index"])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]["index"])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]["index"])[0]

    for i in range(len(scores)):
        if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            w = xmax - xmin
            h = ymax - ymin

            rectangle(
                frame, (xmin, ymin, xmax - xmin, ymax - ymin), (223, 65, 80)[::-1], 2
            )

            object_name = labels[int(classes[i])]
            label = f"{object_name} {int(scores[i] * 100)}%"

            ox, oy = max(0, xmin), max(35, ymin)
            (w, h), _ = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.7, 1)
            x1, y1, x2, y2 = ox, oy, ox + w, oy - h

            rectangle(frame, (x1, y1), (x2 + 10, y2 - 10), (223, 65, 80)[::-1], FILLED)
            putText(
                frame,
                label,
                (ox + 5, oy - 5),
                FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255)[::-1],
                1,
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
    if is_pressed("q"):
        break
    waitKey(1)

destroyAllWindows()
videostream.stop()
