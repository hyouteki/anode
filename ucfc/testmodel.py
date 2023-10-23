from numpy import asarray, array
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import join
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed as tensorflowRandomSeed
from termcolor import colored
from cv2 import (
    VideoCapture,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_POS_FRAMES,
    resize,
)

SEED = 27
tensorflowRandomSeed(SEED)

# DATASET_NAME = "UCF50"
DATASET_NAME = "UCFCrimeDataset"
allClassNames = listdir(DATASET_NAME)
samplesInEachClass = [
    listdir(join(DATASET_NAME, className)) for className in allClassNames
]

"""
# Reduced frame dimensions
# >> Resizing the video frame dimension to a fixed size
"""
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DIMENSION = (IMAGE_WIDTH, IMAGE_HEIGHT)

SEQUENCE_LENGTH = 20
"""
Extracts a total of `SEQUENCE_LENGTH` number of frames form every video \
    (sample) at every equal interval. 
"""


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


def extractTrainFeaturesAndLabels(trainClasses: list[str]):
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
        print(colored(f"[DEBUG] extracting Data of Class: {className}", "blue"))
        files = listdir(join(DATASET_NAME, className))
        for file in files:
            videoFilePath = join(DATASET_NAME, className, file)
            frames = frameExtraction(videoFilePath)
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(classId)
    features = asarray(features)
    labels = array(labels)
    oneHotEncodedLabels = to_categorical(labels)
    return features, oneHotEncodedLabels


"""
Splitting the features and labels into train and test dataset with \
    `test_size = 0.2` and shuffling enabled.
"""
# trainClasses = ["BenchPress", "CleanAndJerk", "Diving", "BreastStroke"]
trainClasses = allClassNames
features, oneHotEncodedLabels = extractTrainFeaturesAndLabels(trainClasses)
featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
    features, oneHotEncodedLabels, test_size=0.2, shuffle=True, random_state=SEED
)

MODEL_NAME = "DS_UCFCrimeDataset___DT_2023_10_22__20_37_00.h5"

model = load_model(MODEL_NAME)

loss, accuracy = model.evaluate(featuresTest, labelsTest)
print(colored(f"[DEBUG] LOSS = {loss}", "blue"))
print(colored(f"[DEBUG] ACC. = {accuracy}", "blue"))
