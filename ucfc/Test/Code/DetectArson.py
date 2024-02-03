import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

SEQUENCE_LENGTH=92
FPS=30
FRAME_COUNT=FPS*7

def frameReduction(frames):
    skipFrameWindow = max(int(FRAME_COUNT / SEQUENCE_LENGTH), 1)
    return np.array([cv2.resize(np.array(frames[i*skipFrameWindow]),(92,92)) / 255
            for i in range(SEQUENCE_LENGTH)])


model = load_model('C:\\Users\\91881\\Documents\\BTP\\Models\\DS_UCFCrimeDataset___C_Arson___DT_2023_11_22__22_40_51.h5')
path = "C:\\Users\\91881\\Documents\\BTP\\Video\\Arson\\Arson020_x264.mp4"
# dir_list = os.listdir(path)
anomalyDetected=0

cap = cv2.VideoCapture(path)
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print( length )
frame_sequence = []
i=0
seconds=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        print(ret)
        
        if(len(frame_sequence)>=FRAME_COUNT):
            frames = frameReduction(frame_sequence)
            anomaly_scores=model.predict(np.array([frames]))
            print(anomaly_scores)
            print("Hello")
            avg_anomaly_score=np.mean(anomaly_scores)
            print(avg_anomaly_score)
            if avg_anomaly_score > 0.5:  # Set the threshold according to your model and problem
                print('Anomaly detected!')
                anomalyDetected+=1
                break
        else:
            frame_sequence.append(frame)

    else:
        break
    # if(cv2.waitKey()==ord('q')):
    #     break
    seconds+=1
    if(seconds%FPS==0):
        print('seconds= ',seconds//FPS)

    # Release the video file
cap.release()

# print(len(dir_list)==anomalyDetected)
print(anomalyDetected)