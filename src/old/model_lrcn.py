from parameters import *
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

tensorflowRandomSeed(SEED)

"""
Splitting the features and labels into train and test dataset with \
    `test_size = TRAIN_TEST_SPLIT` and shuffling enabled.
"""
features = numpyLoad("../npyfiles/features.npy")
oneHotEncodedLabels = numpyLoad("../npyfiles/1hotenclab.npy")
print(colored(f"[DEBUG] Features shape: {features.shape}", "blue"))
print(colored(f"[DEBUG] 1HotEncL shape: {oneHotEncodedLabels.shape}", "blue"))
featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
    features, 
    oneHotEncodedLabels, 
    test_size=TRAIN_TEST_SPLIT, 
    shuffle=True, 
    random_state=SEED
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
    model.add(Dense(len(TRAIN_CLASSES), activation="softmax"))

    return model

model = createModelArchitecture()

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
model.save(f"../models/DS_{DATASET_NAME}___DT_{currentDateTime}.h5")

loss, accuracy = model.evaluate(featuresTest, labelsTest)
print(colored(f"[RESULT] LOSS = {loss}", "green"))
print(colored(f"[RESULT] ACC. = {accuracy}", "green"))
