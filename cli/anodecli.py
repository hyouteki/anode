import time
import sys
import cv2
import os
import numpy as np
import pickle
import tensorflow as tf
from parameters import *

MODELS = {
    "arson": "Arson92.tflite",
    "explosion": "Explosion.tflite", 
    "road_accidents": "RoadAccidents.tflite",
    "shooting": "Shooting.tflite",
    "vandalism": "Vandalism.tflite",
    "fighting": "Fighting.tflite",
}

FRAME_COUNT = SEQUENCE_LENGTH
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

    predictions = 0
    
    interpreter = tf.lite.Interpreter(model_path=os.path.join("models", MODELS[model_name]))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    def current_milli_time():
        return round(time.time() * 1000)
    
    def predict_output(frames):
        start_time = current_milli_time()
        interpreter.set_tensor(input_details[0]['index'], frames)
        interpreter.invoke()
        end_time = current_milli_time()
        print(f"\tPrediction_time: {(end_time-start_time)/1000:.4f} secs")
        sys.stdout.flush()
        return interpreter.get_tensor(output_details[0]['index'])
    
    # exit(0)
    buffer = []
    video_capture = cv2.VideoCapture(video_path)
    
    i = 0
    anomaly_scores = []
    normal_scores = []
    start = current_milli_time()
    fps = video_capture.get(cv2.CAP_PROP_FPS)
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
        prediction = predict_output(np.array([frames], dtype=np.float32))[0]
        print(f"\tPrediction: {prediction}")
        anomaly_scores.append(prediction[0])
        normal_scores.append(prediction[1])
        buffer = buffer[FRAME_COUNT//3: ]
        predictions += 1
    end = current_milli_time()
    time_taken = (end-start)/1000
    video_length = i/fps
    print(f"Video_length: {video_length} secs")
    print(f"Time_elapsed: {time_taken:.4f} secs")
    print(f"Time_realtime_ratio: {time_taken/video_length:.4f}")
    print(f"Predictions: {predictions}")
    print(f"Prediction_ratio: {predictions/i:.4f}")
    with open(out_file_name, "wb") as file:
        pickle.dump({"anomaly_scores": anomaly_scores, "normal_scores": normal_scores}, file)
