from parameters import *
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed as tensorflowRandomSeed
from termcolor import colored
import cv2
from skimage import segmentation, color

tensorflowRandomSeed(SEED)

def applyOpticFlow(frames):
    labels = segmentation.slic(image=frames, compactness=20, n_segments=60)
    out = color.label2rgb(labels, frames, kind='avg')
    return out

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
    videoReader = cv2.VideoCapture(videoPath)
    # total number of frames present in the video
    frameCount = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
    skipFrameWindow = max(int(frameCount / SEQUENCE_LENGTH), 1)
    for i in range(SEQUENCE_LENGTH):
        videoReader.set(cv2.CAP_PROP_POS_FRAMES, i * skipFrameWindow)
        success, frame = videoReader.read()
        # if not successful in reading the frame break from the loop
        if not success:
            break
        # append the frame on frames after resizing
        frames.append(cv2.resize(frame, IMAGE_DIMENSION) / 255)
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
        print(colored(f"[DEBUG] extracting Data of Class: {className}", "blue"))
        files = os.listdir(os.path.join(DATASET_PATH, className))

        for file in files:
            videoFilePath = os.path.join(DATASET_PATH, className, file)

            frames = frameExtraction(videoFilePath)
            frames = applyOpticFlow(frames)
            features.append(frames)

            labels.append(classId)

    features = np.asarray(features)
    labels = np.array(labels)
    oneHotEncodedLabels = to_categorical(labels)
    return features, oneHotEncodedLabels

features, oneHotEncodedLabels = extractFeaturesAndLabels(TRAIN_CLASSES)

print(colored(f"[DEBUG] Features shape: {features.shape}", "magenta"))
print(colored(f"[DEBUG] 1HotEncL shape: {oneHotEncodedLabels.shape}", "magenta"))
np.save("npyfiles/features_optic.npy", features)
np.save("npyfiles/1hotenclab_optic.npy", oneHotEncodedLabels)
