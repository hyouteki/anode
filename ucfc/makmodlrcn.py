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
    TimeDistributed,
    Dropout,
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    LSTM,
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
    model.add(Dense(10, activation="softmax"))

    return model


model = createModelArchitecture()


"""
- Reference for early stopping: \
    https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/
"""
# Create an Instance of Early Stopping Callback.
earlyStoppingCallback = EarlyStopping(
    monitor="val_loss",
    patience=15,
    mode="min",
    restore_best_weights=True,
)
# Compile the model and specify loss function, optimizer and metrics to the model.
model.compile(
    loss="categorical_crossentropy",
    optimizer="Adam",
    metrics=["accuracy"],
)

# Start training the model.
modelTrainingHistory = model.fit(
    x=featuresTrain,
    y=labelsTrain,
    epochs=100,
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
