from parameters import *
from numpy import asarray, array
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import join
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed as tensorflowRandomSeed
import datetime as dt
from tensorflow.keras.layers import (
    TimeDistributed,
    Dropout,
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    LSTM,
)
from cv2 import (
    VideoCapture,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_POS_FRAMES,
    resize,
)

tensorflowRandomSeed(SEED)

def frameExtraction(videoPath):
    """
    Extracts frames from the video at videoPath

    Parameters
    ----------
    - videoPath : str
        - path of the video
    
    Returns
    -------
    - frames : list 
        - `SEQUENCE_LENGTH` number of frames that are equally spaced out \
            in the video.
    """
    frames = []
    videoReader = VideoCapture(videoPath)
    # total number of frames present in the video
    frameCount = int(videoReader.get(CAP_PROP_FRAME_COUNT))
    skipFrameWindow = max(int(frameCount / SEQUENCE_LENGTH), 1)
    for i in range(SEQUENCE_LENGTH):
        videoReader.set(CAP_PROP_POS_FRAMES, i * skipFrameWindow)
        success, frame = videoReader.read()
        # if not successful in reading the frame break from the loop
        if not success:
            break
        # append the frame on frames after resizing
        frames.append(resize(frame, IMAGE_DIMENSION) / 255)
    videoReader.release()
    return frames


def extractFeaturesAndLabels(trainClasses):
    """
    Extracting features and labels from `CLASSES` (train classses)

    PARAMETERS
    ----------
    - trainClasses : list[str]
        - Classes on which the model currently being trained upon maybe \
            equal to `allClassNames`.

    RETURNS
    -------
    - features : 2D list
        - vector of feature (vector of frame in a video)
    - oneHotEncodedLabels : list[list[int]]
        - vector of hotEncodedLabel corresponding to a feature.
        - Ex. [1 0 0 0] : meaning that the corresponding feature belongs to class[0]

    """
    features, labels = [], []
    for classId, className in enumerate(trainClasses):
        print(f"Extracting for class: {className}")
        files = listdir(join(DATASET_PATH, className))
        for file in files:
            videoFilePath = join(DATASET_PATH, className, file)
            features.append(frameExtraction(videoFilePath))
            labels.append(classId)
    features = asarray(features)
    labels = array(labels)
    oneHotEncodedLabels = to_categorical(labels)
    return features, oneHotEncodedLabels


def createModelArchitecture(trainClasses):
    """
    Creates a LRCN model

    RETURNS
    -------
    - model : Sequential
    """

    model = Sequential()
    model.add(
        TimeDistributed(
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        )
    )

    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu")))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu")))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu")))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    # model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))
    model.add(Dense(len(trainClasses), activation="softmax"))

    return model

def makeModelForIndividualClass(trainClasses):
    features, oneHotEncodedLabels = extractFeaturesAndLabels(trainClasses)
    featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
        features, 
        oneHotEncodedLabels, 
        test_size=TRAIN_TEST_SPLIT, 
        shuffle=True, 
        random_state=SEED
    )
    
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
    currentDateTime = dt.datetime.strftime(dt.datetime.now(), "%Y_%m_%d__%H_%M_%S")
    model.save(f"../models/individual/DS_{DATASET_NAME}___C_{trainClasses[0]}___DT_{currentDateTime}.h5")
    loss, accuracy = model.evaluate(featuresTest, labelsTest)
    with open("ind_lrcn_obs.md", "a") as file:
        file.write(f"## {trainClasses[0]}")
        file.write(f"- LOSS = {loss}")
        file.write(f"- ACC. = {accuracy}")
        print(f"## {trainClasses[0]}")
        print(f"- LOSS = {loss}")
        print(f"- ACC. = {accuracy}")

for trainClass in TRAIN_CLASSES[:-1]:
    makeModelForIndividualClass([trainClass, "Normal"])
