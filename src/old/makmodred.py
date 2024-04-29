from numpy import load as numpyLoad
from sklearn.model_selection import train_test_split
from os import listdir
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed as tensorflowRandomSeed
from termcolor import colored
import datetime as dt
from tensorflow.keras.layers import (
    ConvLSTM2D,
    MaxPooling3D,
    TimeDistributed,
    Dropout,
    Flatten,
    Dense,
)

SEED = 27
tensorflowRandomSeed(SEED)

DATASET_NAME = "UCFCrimeDataset"
allClassNames = listdir(DATASET_NAME)
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


def getClassIdByName(_className):
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


"""
Splitting the features and labels into train and test dataset with \
    `test_size = 0.2` and shuffling enabled.
"""
trainClasses = uniqueClassName
features = numpyLoad("npyfiles/features.npy")
oneHotEncodedLabels = numpyLoad("npyfiles/1hotenclab.npy")
print(colored(f"[DEBUG] Features shape: {features.shape}", "blue"))
print(colored(f"[DEBUG] 1HotEncL shape: {oneHotEncodedLabels.shape}", "blue"))
featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
    features, oneHotEncodedLabels, test_size=0.2, shuffle=True, random_state=SEED
)


def createModelArchitecture():
    """
    Model: "sequential"

    | Layer (type)                          | Output Shape              | Param |
    | :-----------                          | :-----------              | :---- |
    | conv_lstm2d (ConvLSTM2D)              | (None, 20, 62, 62, 4)     | 1024  |
    | max_pooling3d (MaxPooling3D)          | (None, 20, 31, 31, 4)     | 0     |
    | time_distributed (TimeDistributed)    | (None, 20, 31, 31, 4)     | 0     |
    | conv_lstm2d_1 (ConvLSTM2D)            | (None, 20, 29, 29, 8)     | 3488  |
    | max_pooling3d_1 (MaxPooling3D)        | (None, 20, 15, 15, 8)     | 0     |
    | time_distributed_1 (TimeDistributed)  | (None, 20, 15, 15, 8)     | 0     |
    | conv_lstm2d_2 (ConvLSTM2D)            | (None, 20, 13, 13, 14)    | 11144 |
    | max_pooling3d_2 (MaxPooling3D)        | (None, 20, 7, 7, 14)      | 0     |
    | time_distributed_2 (TimeDistributed)  | (None, 20, 7, 7, 14)      | 0     |
    | conv_lstm2d_3 (ConvLSTM2D)            | (None, 20, 5, 5, 16)      | 17344 |
    | max_pooling3d_3 (MaxPooling3D)        | (None, 20, 3, 3, 16)      | 0     |
    | flatten (Flatten)                     | (None, 2880)              | 0     |
    | dense (Dense)                         | (None, 4)                 | 11524 |

    - Total params: 44524 (173.92 KB)
    - Trainable params: 44524 (173.92 KB)
    - Non-trainable params: 0 (0.00 Byte)

    RETURNS
    -------
    - model : Sequential
    """
    model = Sequential(
        [
            ConvLSTM2D(
                filters=4,
                kernel_size=(3, 3),
                activation="tanh",
                data_format="channels_last",
                recurrent_dropout=0.2,
                return_sequences=True,
                input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            ),
            MaxPooling3D(
                pool_size=(1, 2, 2), padding="same", data_format="channels_last"
            ),
            TimeDistributed(Dropout(0.2)),
            ConvLSTM2D(
                filters=8,
                kernel_size=(3, 3),
                activation="tanh",
                data_format="channels_last",
                recurrent_dropout=0.2,
                return_sequences=True,
            ),
            MaxPooling3D(
                pool_size=(1, 2, 2), padding="same", data_format="channels_last"
            ),
            TimeDistributed(Dropout(0.2)),
            ConvLSTM2D(
                filters=14,
                kernel_size=(3, 3),
                activation="tanh",
                data_format="channels_last",
                recurrent_dropout=0.2,
                return_sequences=True,
            ),
            MaxPooling3D(
                pool_size=(1, 2, 2), padding="same", data_format="channels_last"
            ),
            TimeDistributed(Dropout(0.2)),
            ConvLSTM2D(
                filters=16,
                kernel_size=(3, 3),
                activation="tanh",
                data_format="channels_last",
                recurrent_dropout=0.2,
                return_sequences=True,
            ),
            MaxPooling3D(
                pool_size=(1, 2, 2), padding="same", data_format="channels_last"
            ),
            Flatten(),
            Dense(len(trainClasses), activation="softmax"),
        ]
    )
    return model


model = createModelArchitecture()

# """
# - Reference for early stopping: \
#     https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/
# """
earlyStoppingCallback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode="min",
    baseline=None,
    restore_best_weights=True,
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
modelTrainingHistory = model.fit(
    x=featuresTrain,
    y=labelsTrain,
    epochs=10000,
    batch_size=15,
    shuffle=True,
    validation_split=0.2,
    callbacks=[earlyStoppingCallback],
)

currentDateTime = dt.datetime.strftime(dt.datetime.now(), "%Y_%m_%d__%H_%M_%S")
model.save(f"DS_{DATASET_NAME}___DT_{currentDateTime}.h5")

loss, accuracy = model.evaluate(featuresTest, labelsTest)
print(colored(f"[RESULT] LOSS = {loss}", "green"))
print(colored(f"[RESULT] ACC. = {accuracy}", "green"))
