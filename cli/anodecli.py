import sys
import cv2
import os
import numpy as np
import pickle
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from parameters import *

MODELS = {
    "arson": "DS_UCFCrimeDataset___C_Arson___DT_2023_11_22__22_40_51.h5",
    "explosion": "DS_UCFCrimeDataset___C_Explosion___DT_2023_11_22__23_22_08.h5", 
    "road_accidents": "DS_UCFCrimeDataset___C_RoadAccidents___DT_2023_11_22__23_03_08.h5",
    "shooting": "DS_UCFCrimeDataset___C_Shooting___DT_2023_11_22__23_57_57.h5",
    "vandalism": "DS_UCFCrimeDataset___C_Vandalism___DT_2023_11_22__23_39_07.h5",
}

SKIP_FRAME_WINDOW = max(int(FRAME_COUNT / SEQUENCE_LENGTH), 1)

def resize_frame(frame):
    return cv2.resize(frame, IMAGE_DIMENSION) / 255

def reduce_buffer(frames):
    return [frames[i * SKIP_FRAME_WINDOW] for i in range(SEQUENCE_LENGTH)]

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("usage: anodecli -m <model> -v <videopath> -o <predictions.pickle>")
        exit(1)

    model_name = sys.argv[2]
    video_path = sys.argv[4]
    out_file_name = sys.argv[6]

    model = load_model(os.path.join("models", MODELS[model_name]))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    
    buffer = []
    video_capture = cv2.VideoCapture(video_path)

    i = 0
    anomaly_scores = []
    normal_scores = []
    while video_capture.isOpened():
        i += 1
        print(f"Frame: {i}")
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
        prediction = model.predict(np.array([frames]))[0]
        print(prediction)
        anomaly_scores.append(prediction[0])
        normal_scores.append(prediction[1])
    with open(out_file_name, "wb") as file:
        pickle.dump({"anomaly_scores": anomaly_scores, "normal_scores": normal_scores}, file)
