from parameters import *
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed as tensorflowRandomSeed
from tensorflow.keras.layers import (TimeDistributed, Dropout, Flatten, Dense,
                                     Conv2D, MaxPooling2D, LSTM)

tensorflowRandomSeed(SEED)

SEQUENCE_LENGTH = 39
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_DIMENSION = (IMAGE_HEIGHT, IMAGE_WIDTH)

def computeOpticalFlow(lastFrame, frame):
    return cv2.calcOpticalFlowFarneback(lastFrame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def extractOpticalFlowFeatures(flow):
    return [np.mean(flow), np.std(flow), np.max(flow), np.min(flow)]   

def preprocessFrame(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), IMAGE_DIMENSION) / 255

def frameExtraction(videoPath):
    frames = []
    videoReader = cv2.VideoCapture(videoPath)
    frameCount = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
    skipFrameWindow = max(int(frameCount/SEQUENCE_LENGTH), 1)
    accumulatedFlow = np.zeros((*IMAGE_DIMENSION, 2))
    
    for i in range(SEQUENCE_LENGTH):
        videoReader.set(cv2.CAP_PROP_POS_FRAMES, i*skipFrameWindow)
        success, frame = videoReader.read()
        if not success:
            break
        frames.append(preprocessFrame(frame))
        if len(frames) == 1:
            continue
        accumulatedFlow += computeOpticalFlow(frames[-2], frames[-1])
    videoReader.release()

    accumulatedFlow = np.transpose(accumulatedFlow, (2, 0, 1))
    frames.extend(accumulatedFlow)
    return frames[1: ]

def extractFeaturesAndLabels(className, classId):
    features, labels = [], []
    from tqdm import tqdm
    print(f"Debug: extracting data for class '{className}'")
    files = os.listdir(os.path.join(DATASET_PATH, className))
    for file in tqdm(files):
        videoFilePath = os.path.join(DATASET_PATH, className, file)
        features.append(frameExtraction(videoFilePath))
        labels.append(classId)
    features = np.asarray(features)
    labels = np.array(labels)
    return features, labels

def createModelArchitecture(trainClasses):
    inputShape = (SEQUENCE_LENGTH+1, *IMAGE_DIMENSION, 1)
    conv2dParams = {"padding": "same", "activation": "relu"}
    model = Sequential()

    model.add(TimeDistributed(Conv2D(16, (3, 3), **conv2dParams), input_shape=inputShape))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), **conv2dParams)))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), **conv2dParams)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), **conv2dParams)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))
    model.add(Dense(len(trainClasses), activation="softmax"))

    return model

def makeModelForIndividualClass(trainClass, normalFeatures, normalLabels):
    features, labels = extractFeaturesAndLabels(trainClass, 1)
    features = np.concatenate((features, normalFeatures), axis=0)
    features = np.expand_dims(features, axis=-1)
    labels = np.concatenate((labels, normalLabels), axis=0)
    oneHotEncodedLabels = to_categorical(labels)
    featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
        features, oneHotEncodedLabels, test_size=TRAIN_TEST_SPLIT, shuffle=True, random_state=SEED)

    print(features.shape, featuresTrain.shape, oneHotEncodedLabels.shape, labelsTrain.shape)
    trainClasses = ["Normal", trainClass]
    model = createModelArchitecture(trainClasses)

    # """
    # - Reference for early stopping: \
    #     https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/
    # """
    earlyStoppingCallback = EarlyStopping(
        monitor=EARLY_STOPPING_CALLBACK_MONITOR,
        min_delta=EARLY_STOPPING_CALLBACK_MIN_DELTA,
        patience=EARLY_STOPPING_CALLBACK_PATIENCE,
        verbose=1,
        mode="min",
        baseline=None,
        restore_best_weights=True,
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    modelTrainingHistory = model.fit(
        x=featuresTrain,
        y=labelsTrain,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=TRAIN_VALID_SPLIT,
        callbacks=[earlyStoppingCallback],
    )
    model.save(f"../models/individual/opticalflow/{trainClass}")
    loss, accuracy = model.evaluate(featuresTest, labelsTest)
    with open("OF-CNNLSTM_obs.md", "a") as file:
        file.write(f"## {trainClass}\n")
        file.write(f"- LOSS = {loss}\n")
        file.write(f"- ACC. = {accuracy}\n")
        print(f"## {trainClass}")
        print(f"|\tLOSS - {loss}")
        print(f"|\tACC. - {accuracy}")


normalFeatures, normalLabels = extractFeaturesAndLabels("Normal", 0)
for trainClass in TRAIN_CLASSES[3: -1]:
    makeModelForIndividualClass(trainClass, normalFeatures, normalLabels)
