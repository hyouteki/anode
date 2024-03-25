import sys
import cv2
import os
import random
import numpy as np
import tensorflow as tf
from parameters import *

TEST_DATASET = [
    (True, "../Fighting/dataset/CCTV_DATA/testing/"),
    (True, "../Fighting/dataset/NON_CCTV_DATA/testing/"),
    (True, "../UCFCrimeDataset/Fighting2/"),
    (False, "../UCFCrimeDataset/Normal/"),
]

FRAME_COUNT = SEQUENCE_LENGTH
SKIP_FRAME_WINDOW = max(int(FRAME_COUNT / SEQUENCE_LENGTH), 1)

def resize_frame(frame):
    return cv2.resize(frame, IMAGE_DIMENSION) / 255

def reduce_buffer(frames):
    return [frames[i * SKIP_FRAME_WINDOW] for i in range(SEQUENCE_LENGTH)]

interpreter = tf.lite.Interpreter(model_path="Fighting.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_output(frames):
    interpreter.set_tensor(input_details[0]['index'], frames)
    interpreter.invoke()
    sys.stdout.flush()
    return interpreter.get_tensor(output_details[0]['index'])

def anomaly_present(video_path):
    past_predictions = []
    buffer = []
    video_capture = cv2.VideoCapture(video_path)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            video_capture.release()
            break
        
        buffer.append(resize_frame(frame))
        if len(buffer) < FRAME_COUNT:
            continue
        if len(buffer) > FRAME_COUNT:
            buffer = buffer[1:]

        frames = reduce_buffer(buffer)
        prediction = predict_output(np.array([frames], dtype=np.float32))[0]
        buffer = buffer[FRAME_COUNT//3: ]

        past_predictions.append(prediction)
        if len(past_predictions) == 3:
            anomaly = True
            for p in past_predictions:
                if p[0] <= p[1]:
                    anomaly = False
                    break
                
            # if prob(anomaly) > prob(normal) for atleast 3 consecutive predictions
            past_predictions = past_predictions[1:]
            if anomaly:
                return True        

    return False


if __name__ == "__main__":
    random.shuffle(TEST_DATASET)
    count = 0
    correct = 0
    for dataset in TEST_DATASET:
        actual_value = dataset[0]
        dataset_loc = dataset[1]
        print(f"DATASET: {dataset_loc}")
        for video in os.listdir(dataset_loc):
            count += 1
            video_path = os.path.join(dataset_loc, video)
            if actual_value == anomaly_present(video_path):
                correct += 1
            print(f"CORRECT: {correct} | COUNT: {count} | ACC: {correct/count:.4f}")
