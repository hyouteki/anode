from parameters import *
from numpy import asarray, array
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import join
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
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
    BatchNormalization,
    AveragePooling2D,
    Activation,
    LSTM,
)
from cv2 import (
    VideoCapture,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_POS_FRAMES,
    resize,
)

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_DIMENSION = (IMAGE_HEIGHT, IMAGE_WIDTH)

def createModelArchitecture():
    return Sequential([LSTM(units=32, input_shape=(SEQUENCE_LENGTH+1, 2), return_sequences=False),
                       Dense(2, use_bias=True, activation="softmax")])


model = createModelArchitecture()

model.summary()
