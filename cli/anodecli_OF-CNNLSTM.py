import time
import sys
import cv2
import numpy as np
import pickle
import tensorflow as tf
from termcolor import colored
from parameters import *

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
    if len(sys.argv) != 4:
       print("usage: anodecli <model> <videopath> <predictions.pkl>")
       exit(1)

    modelName = sys.argv[1]
    videoPath = sys.argv[2]
    outFileName = sys.argv[3]

    numPredictions = 0
    predictions = []
    interpreter = tf.lite.Interpreter(model_path=f"../models/OF-CNNLSTM/{modelName}.tflite")
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
    frameCount = 0
    start = currentMilliTime()
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    
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

        frames = preprocessFrames(buffer)
        prediction = predictOutput(np.array([frames], dtype=np.float32))[0]
        predictions.append(prediction)
        print(colored(f"NormalScore({prediction[0]:.5f}), AnomalyScore({prediction[1]:.5f})", "green"))
        buffer = buffer[FRAME_COUNT//WINDOW_SCALE: ]
        numPredictions += 1
        print(colored(f"TimeElapsed({frameCount/fps:.4f} secs)", "green"))

        if len(predictions) >= 5:
            anomaly = True
            for prediction in predictions[-5: ]:
                if prediction[0] >= prediction[1] or prediction[1] <= 0.65:
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
