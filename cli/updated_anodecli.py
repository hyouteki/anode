import time
import sys
import cv2
import os
import numpy as np
import pickle
import tensorflow as tf
from termcolor import colored
from parameters import *

MODELS = {
    "arson": "Arson.tflite",
    "explosion": "Explosion.tflite", 
    "road_accidents": "RoadAccidents.tflite",
    "shooting": "Shooting.tflite",
    "vandalism": "Vandalism.tflite",
    "fighting": "Fighting.tflite",
}

ENABLE_BUZZER = False
if ENABLE_BUZZER:
    from gpiozero import Buzzer
    buzzer = Buzzer(17)

ENABLE_IMSHOW = False
    
WINDOW_SCALE = 8
SEQUENCE_LENGTH = 39
FRAME_COUNT = SEQUENCE_LENGTH
IMAGE_DIMENSION = (128, 128)
SKIP_FRAME_WINDOW = max(int(FRAME_COUNT / SEQUENCE_LENGTH), 1)

def computeOpticalFlow(lastFrame, frame):
    return cv2.calcOpticalFlowFarneback(lastFrame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def preprocessFrames(frames):
    accumulatedFlow = np.zeros((*IMAGE_DIMENSION, 2))
    for i, frame in enumerate(frames[1: ]):
        lastFrame = frames[i-1]
        accumulatedFlow += computeOpticalFlow(lastFrame, frame)
    accumulatedFlow = np.transpose(accumulatedFlow, (2, 0, 1))
    frames = frames[1: ]
    frames.extend(accumulatedFlow)
    return np.expand_dims(frames, axis=-1)

if __name__ == "__main__":
    if len(sys.argv) < 7:
       print("usage: anodecli -m <model> -v <videopath> -o <predictions.pickle>")
       exit(1)

    modelName = sys.argv[2]
    videoPath = sys.argv[4]
    outFileName = sys.argv[6]

    predictions = 0
    pastPredictions = []
    interpreter = tf.lite.Interpreter(model_path=os.path.join("models", MODELS[modelName]))
    interpreter.allocate_tensors()
    inputDetails = interpreter.get_input_details()
    outputDetails = interpreter.get_output_details()
    print(inputDetails)
    print(outputDetails)

    def currentMilliTime():
        return round(time.time() * 1000)
    
    def predictOutput(frames):
        startTime = currentMilliTime()
        interpreter.set_tensor(inputDetails[0]['index'], frames)
        interpreter.invoke()
        endTime = currentMilliTime()
        print(colored(f"PredictionTime({(endTime-startTime)/1000:.4f} secs)", "green"))
        sys.stdout.flush()
        return interpreter.get_tensor(outputDetails[0]['index'])
    
    buffer = []
    videoCapture = cv2.VideoCapture(0 if videoPath == "cam" else videoPath)    
    i = 0
    anomalyScores = []
    normalScores = []
    start = currentMilliTime()
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    
    while videoCapture.isOpened():
        i += 1
        print(f"Frame({i})")

        success, frame = videoCapture.read()
        if not success:
            videoCapture.release()
            break
        if ENABLE_IMSHOW:
            cv2.imshow("Frame", frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, IMAGE_DIMENSION) / 255

        buffer.append(frame)
        if len(buffer) < FRAME_COUNT:
            continue

        frames = preprocessFrames(buffer)
        prediction = predictOutput(np.array([frames], dtype=np.float32))[0]
        print(colored(f"Prediction({prediction})", "green"))
        anomalyScores.append(prediction[0])
        normalScores.append(prediction[1])
        buffer = buffer[FRAME_COUNT//WINDOW_SCALE: ]
        predictions += 1
        print(colored(f"TimeElapsed({i/fps:.4f} secs)", "green"))

        pastPredictions.append(prediction)
        if len(pastPredictions) == 5:
            anomaly = True
            for p in pastPredictions:
                if p[0] <= p[1] or p[0] <= 0.65:
                    anomaly = False
                    break
                
            pastPredictions = pastPredictions[1:]
            if anomaly:
                if ENABLE_BUZZER:
                    buzzer.on()
                print(colored("<=============================>", "red"))
                print(colored("<=========> ANOMALY <=========>", "red"))
                print(colored("<=============================>", "red"))
            else:
                if ENABLE_BUZZER:
                    buzzer.off()
                    
        predictions += 1
        
    end = currentMilliTime()
    timeTaken = (end-start)/1000
    videoLength = i/fps
    print(f"VideoLength({videoLength} secs), TimeElapsed({timeTaken:.4f} secs)")
    print(f"TimeRealtimeRatio({timeTaken/videoLength:.4f})")
    print(f"NumPredictions({predictions}), PredictionRatio({predictions/i:.4f})")
    with open(outFileName, "wb") as file:
        pickle.dump({"anomalyScores": anomalyScores, "normalScores": normalScores}, file)
