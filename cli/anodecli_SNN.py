import time
import sys
import cv2
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from termcolor import colored
from parameters import *

import sys
sys.path.append("../src/spkeras/")
from spkeras.models import cnn_to_snn

ENABLE_BUZZER = False
if ENABLE_BUZZER:
    from gpiozero import Buzzer
    buzzer = Buzzer(17)

ENABLE_IMSHOW = True
    
WINDOW_SCALE = 8
SEQUENCE_LENGTH = 39
FRAME_COUNT = SEQUENCE_LENGTH
IMAGE_DIMENSION = (128, 128)
SKIP_FRAME_WINDOW = max(int(FRAME_COUNT / SEQUENCE_LENGTH), 1)

def currentMilliTime():
    return round(time.time() * 1000)

class MyModel:
    def __init__(self, modelName, architecture):
        cnnModel = load_model(f"../models/OF-CNN/{modelName}.cnn.h5")

        self.architecture = architecture
        if architecture == "OF-CNN":
            self.model = cnnModel
            return

        print("Debug: Extracting featuresTrain from ds cache")
        normalFeatures, normalLabels = pickle.load(open(f"../ds/ucfc/Normal.cache.pkl", "rb"))
        features, labels = pickle.load(open(f"../ds/ucfc/{modelName}.cache.pkl", "rb"))        
        features = np.concatenate((features, normalFeatures), axis=0)
        features = np.expand_dims(features, axis=-1)
        labels = np.concatenate((labels, normalLabels), axis=0)
        featuresTrain = train_test_split(features, labels, test_size=TRAIN_TEST_SPLIT,
                                         shuffle=True, random_state=SEED)[0]
        print(f"Debug: Started building SNN({modelName})")
        self.model = cnn_to_snn(signed_bit=0)(cnnModel, featuresTrain)
        if architecture == "OF-SNN":
            return
        elif architecture == "OF-SNNLSTM":
            self.modelLstm = load_model(f"models/OF-SNNLSTM/{modelName}.snn-lstm.h5")
        else:
            print("Error: invalid architecture expected(OF-CNN, OF-SNN, OF-SNNLSTM)")
            exit(1)
            
    def predict(self, features):
        startTime = currentMilliTime()
        predictions = self.model.predict(features)
        predictions = self.modelLstm.predict(predictions) if architecture == "OF-SNNLSTM" \
            else [np.mean(predictions[:, 0]), np.mean(predictions[:, 1])]
        endTime = currentMilliTime()
        print(colored(f"PredictionTime({(endTime-startTime)/1000:.4f} secs)", "green"))
        return predictions

def computeOpticalFlow(lastFrame, frame):
    return cv2.calcOpticalFlowFarneback(lastFrame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def preprocessFrame(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), IMAGE_DIMENSION) / 255

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
    if len(sys.argv) < 5:
        print("usage: anodecli <model> <architecture> <videopath> <predictions.pkl>")
        exit(1)

    modelName = sys.argv[1]
    architecture = sys.argv[2]
    videoPath = sys.argv[3]
    outFileName = sys.argv[4]

    numPredictions = 0
    predictions = []
    
    buffer = []
    videoCapture = cv2.VideoCapture(0 if videoPath == "cam" else videoPath)    
    frameCount = 0
    start = currentMilliTime()
    fps = 30 if (fps := videoCapture.get(cv2.CAP_PROP_FPS)) == 0 else fps
    model = MyModel(modelName, architecture)
    
    while videoCapture.isOpened():
        frameCount += 1
        print(f"Frame({frameCount})")

        success, frame = videoCapture.read()
        if not success:
            videoCapture.release()
            break
        if ENABLE_IMSHOW:
            cv2.imshow("Frame", frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

        frame = preprocessFrame(frame)
    
        buffer.append(frame)
        if len(buffer) < FRAME_COUNT:
            continue

        prediction = model.predict(preprocessFrames(buffer))
        predictions.append(prediction)
        numPredictions += 1
        buffer = buffer[FRAME_COUNT//WINDOW_SCALE: ]
        
        print(colored(f"NormalScore({prediction[0]:.5f}), AnomalyScore({prediction[1]:.5f})", "green"))
        print(colored(f"TimeElapsed({frameCount/fps:.4f} secs)", "green"))

        checkPastPredictionCount = 5
        if len(predictions) >= checkPastPredictionCount:
            anomaly = True
            for prediction in predictions[-checkPastPredictionCount: ]:
                if prediction[0] <= prediction[1] or prediction[0] <= 0.65:
                    anomaly = False
                    break
                
            if anomaly:
                if ENABLE_BUZZER:
                    buzzer.on()
                print(colored("<=============================>", "red"))
                print(colored("<=========> ANOMALY <=========>", "red"))
                print(colored("<=============================>", "red"))
            else:
                if ENABLE_BUZZER:
                    buzzer.off()

    end = currentMilliTime()
    timeTaken = (end-start)/1000
    videoLength = frameCount/fps
    print(f"VideoLength({videoLength} secs), TimeElapsed({timeTaken:.4f} secs)")
    print(f"TimeRealtimeRatio({timeTaken/videoLength:.4f})")
    print(f"NumPredictions({numPredictions}), PredictionRatio({numPredictions/frameCount:.4f})")
    with open(outFileName, "wb") as file:
        pickle.dump(predictions, file)
