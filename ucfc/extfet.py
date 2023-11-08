from numpy import asarray, array, save as numpySave
from os import listdir
from os.path import join
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

DATASET_NAME = "UCFCrimeDataset"
allClassNames = listdir(DATASET_NAME)


def getClassIdByName(_className):
    uniqueClassName = [
        "Abuse",
        "Arrest",
        "Arson",
        "Fighting",
        "Stealing",
        "Explosion",
        "RoadAccidents",
        "Shooting",
        "Vandalism",
        "Normal",
    ]
    mappingClassName2ClassName = {
        "Abuse": "Abuse",
        "Arrest": "Arrest",
        "Arson": "Arson",
        "Assault": "Fighting",
        "Burglary": "Stealing",
        "Explosion": "Explosion",
        "Fighting": "Fighting",
        "RoadAccidents": "RoadAccidents",
        "Robbery": "Stealing",
        "Shooting": "Shooting",
        "Shoplifting": "Stealing",
        "Stealing": "Stealing",
        "Vandalism": "Vandalism",
        "Normal": "Normal",
    }
    return uniqueClassName.index(mappingClassName2ClassName[_className])


"""
# Reduced frame dimensions
# >> Resizing the video frame dimension to a fixed size
"""
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DIMENSION = (IMAGE_WIDTH, IMAGE_HEIGHT)

SEQUENCE_LENGTH = 40
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
    for className in trainClasses:
        print(colored(f"[DEBUG] extracting Data of Class: {className}", "blue"))
        files = listdir(join(DATASET_NAME, className))
        for file in files:
            videoFilePath = join(DATASET_NAME, className, file)
            features.append(frameExtraction(videoFilePath))
            labels.append(getClassIdByName(className))
    features = asarray(features)
    labels = array(labels)
    oneHotEncodedLabels = to_categorical(labels)
    return features, oneHotEncodedLabels


trainClasses = allClassNames
features, oneHotEncodedLabels = extractFeaturesAndLabels(trainClasses)

print(colored(f"[DEBUG] Features shape: {features.shape}", "magenta"))
print(colored(f"[DEBUG] 1HotEncL shape: {oneHotEncodedLabels.shape}", "magenta"))
numpySave("npyfiles/features.npy", features)
numpySave("npyfiles/1hotenclab.npy", oneHotEncodedLabels)
