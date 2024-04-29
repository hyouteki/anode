from parameters import *
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed as tensorflowRandomSeed
from spkeras.models import cnn_to_snn
import pickle
from tensorflow.keras.layers import (Dropout, Flatten, Dense, Conv2D, AveragePooling2D,
                                     Activation, BatchNormalization)

tensorflowRandomSeed(SEED)

SEQUENCE_LENGTH = 39
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_DIMENSION = (IMAGE_HEIGHT, IMAGE_WIDTH)

EPOCHS = 80

def computeOpticalFlow(lastFrame, frame):
    return cv2.calcOpticalFlowFarneback(lastFrame, frame, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

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

def extractFeaturesAndLabels(className, classId, force=False):
    if os.path.exists(f"{className}.cache.pkl") and not force:
        with open(f"{className}.cache.pkl", 'rb') as file:
            return pickle.load(file)
    features, labels = [], []
    from tqdm import tqdm
    print(f"Debug: extracting data for class '{className}'")
    files = os.listdir(os.path.join(DATASET_PATH, className))
    for file in tqdm(files):
        videoFilePath = os.path.join(DATASET_PATH, className, file)
        features.extend(frameExtraction(videoFilePath))
        labels.extend([classId]*(SEQUENCE_LENGTH+1))
    features = np.asarray(features)
    labels = np.array(labels)
    with open(f"{className}.cache.pkl", 'wb') as file:
        pickle.dump((features, labels), file)
    return features, labels

def createModelArchitecture(trainClasses):
    inputShape = (*IMAGE_DIMENSION, 1)
    conv2dParams = {"padding": "same", "use_bias": True}
    model = Sequential()
    
    model.add(Conv2D(16, (3, 3), **conv2dParams, input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D((4, 4)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), **conv2dParams))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D((4, 4)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), **conv2dParams))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), **conv2dParams))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    denseParams = {"use_bias": True}
    
    model.add(Flatten())
    
    model.add(Dense(2, **denseParams))
    model.add(Activation("softmax"))
    
    return model

def prepareModelInput(trainClass, normalFeatures, normalLabels):
    features, labels = extractFeaturesAndLabels(trainClass, 1)
    features = np.concatenate((features, normalFeatures), axis=0)
    features = np.expand_dims(features, axis=-1)
    labels = np.concatenate((labels, normalLabels), axis=0)
    oneHotEncodedLabels = to_categorical(labels)
    print(oneHotEncodedLabels.shape)
    return train_test_split(features, oneHotEncodedLabels, test_size=TRAIN_TEST_SPLIT,
                            shuffle=True, random_state=SEED)

def makeModelForIndividualClass(trainClass, normalFeatures, normalLabels, force=False):
    if os.path.exists(f"../models/individual/snn/{trainClass}.cnn.h5") and not force:
        print(f"Info: Cached CNN({trainClass})")
        return load_model(f"../models/individual/snn/{trainClass}.cnn.h5")
    featuresTrain, featuresTest, labelsTrain, labelsTest = \
        prepareModelInput(trainClass, normalFeatures, normalLabels)

    print(featuresTrain.shape, labelsTrain.shape)
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
    model.save(f"../models/individual/snn/{trainClass}.cnn.h5")
    loss, accuracy = model.evaluate(featuresTest, labelsTest)
    print(f"Debug: {trainClass} CNN model constructed")
    with open("OF-CNN.md", "a") as file:
        file.write(f"## {trainClass}\n")
        file.write(f"- LOSS = {loss}\n")
        file.write(f"- ACC. = {accuracy}\n")
        print(f"## {trainClass} CNN")
        print(f"|\tLOSS - {loss}")
        print(f"|\tACC. - {accuracy}")
    return model
        
def applySNN(trainClass, normalFeatures, normalLabels, cnnModel):
    featuresTrain, featuresTest, labelsTrain, labelsTest = \
        prepareModelInput(trainClass, normalFeatures, normalLabels)
    
    trainClasses = ["Normal", trainClass]

    snnModel = cnn_to_snn(signed_bit=0)(cnnModel, featuresTrain)
    print(f"Debug: {trainClass} SNN model constructed")
    loss, accuracy = snnModel.evaluate(featuresTest, labelsTest, timesteps=256)
    sMax, s = snnModel.SpikeCounter(featuresTrain, timesteps=256)
    n = snnModel.NeuronNumbers(mode=0)
    with open("OF-SNN.md", "a") as file:
        file.write(f"## {trainClass}\n")
        file.write(f"- LOSS = {loss}\n")
        file.write(f"-  ACC = {accuracy}\n")
        file.write(f"- sMax = {sMax}\n")
        file.write(f"-    s = {s}\n")
        file.write(f"-    n = {n}\n")
        print(f"## {trainClass} SNN")
        print(f"|\tLOSS - {loss}")
        print(f"|\tACC. - {accuracy}")

normalFeatures, normalLabels = extractFeaturesAndLabels("Normal", 0)
for trainClass in TRAIN_CLASSES[: -1]:
    cnnModel = makeModelForIndividualClass(trainClass, normalFeatures, normalLabels)
    applySNN(trainClass, normalFeatures, normalLabels, cnnModel)
