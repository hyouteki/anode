# Model 1: Sequential ConvLSTM
``` Python
Sequential([
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
    Dense(len(TRAIN_CLASSES), activation="softmax"),
])
```

## Test 1

``` Python
SEED = 666
DATASET_NAME = "UCFCrimeDataset"
# feature Parameters
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
SEQUENCE_LENGTH = 20
# dataset partitions
TRAIN_TEST_SPLIT = 0.2
TRAIN_VALID_SPLIT = 0.2
# early stopping callback parameters
EARLY_STOPPING_CALLBACK_MONITOR = "val_loss"
EARLY_STOPPING_CALLBACK_MIN_DELTA = 0.001
EARLY_STOPPING_CALLBACK_PATIENCE = 10
# optimizer parameters
LEARNING_RATE = 0.001
# training parameters
EPOCHS = 30
BATCH_SIZE = 4
```

- LOSS = 2.99981427192688
- ACC. = 0.21052631735801697

## Test 2
> Increased the batch size from `4` to `15`

``` Python
SEED = 666
DATASET_NAME = "UCFCrimeDataset"
# feature Parameters
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
SEQUENCE_LENGTH = 20
# dataset partitions
TRAIN_TEST_SPLIT = 0.2
TRAIN_VALID_SPLIT = 0.2
# early stopping callback parameters
EARLY_STOPPING_CALLBACK_MONITOR = "val_loss"
EARLY_STOPPING_CALLBACK_MIN_DELTA = 0.001
EARLY_STOPPING_CALLBACK_PATIENCE = 10
# optimizer parameters
LEARNING_RATE = 0.001
# training parameters
EPOCHS = 30
BATCH_SIZE = 15
```

- LOSS = 6.2900004386901855
- ACC. = 0.1315789520740509

### Test 3
> Increased Epochs from `30` to `10000`

``` Python
SEED = 666
DATASET_NAME = "UCFCrimeDataset"
# feature Parameters
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
SEQUENCE_LENGTH = 20
# dataset partitions
TRAIN_TEST_SPLIT = 0.2
TRAIN_VALID_SPLIT = 0.2
# early stopping callback parameters
EARLY_STOPPING_CALLBACK_MONITOR = "val_loss"
EARLY_STOPPING_CALLBACK_MIN_DELTA = 0.001
EARLY_STOPPING_CALLBACK_PATIENCE = 10
# optimizer parameters
LEARNING_RATE = 0.001
# training parameters
EPOCHS = 30
BATCH_SIZE = 15
```

- LOSS = 2.4563605785369873
- ACC. = 0.20000000298023224

### Test 4
> Click [here](models/DS_UCFCrimeDataset___DT_2023_10_29__15_27_45.h5) for model<br>
> Reduced classes from `13` to `9`.<br>
> Increase Sequence length from `20` to `40`

``` Python
SEED = 666
DATASET_NAME = "UCFCrimeDataset"
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
}
# feature Parameters
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
SEQUENCE_LENGTH = 40
# dataset partitions
TRAIN_TEST_SPLIT = 0.2
TRAIN_VALID_SPLIT = 0.2
# early stopping callback parameters
EARLY_STOPPING_CALLBACK_MONITOR = "val_loss"
EARLY_STOPPING_CALLBACK_MIN_DELTA = 0.001
EARLY_STOPPING_CALLBACK_PATIENCE = 10
# optimizer parameters
LEARNING_RATE = 0.001
# training parameters
EPOCHS = 30
BATCH_SIZE = 15
```

- LOSS = 1.852685570716858
- ACC. = 0.36315789818763733

### Test 5
> Click [here](models/DS_UCFCrimeDataset___DT_2023_11_13__23_46_40.h5) for model associated with this test.

``` Python
SEED = 666
DATASET_NAME = "UCFCrimeDataset"
TRAIN_CLASSES = [
    "Arson",
    "RoadAccidents",
    "Explosion",
    "Vandalism",
    "Shooting",
    "Normal",
]
# feature Parameters
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
SEQUENCE_LENGTH = 80
# dataset partitions
TRAIN_TEST_SPLIT = 0.25
TRAIN_VALID_SPLIT = 0.25
# early stopping callback parameters
EARLY_STOPPING_CALLBACK_MONITOR = "val_loss"
EARLY_STOPPING_CALLBACK_MIN_DELTA = 0.001
EARLY_STOPPING_CALLBACK_PATIENCE = 15
# optimizer parameters
LEARNING_RATE = 0.001
# training parameters
EPOCHS = 250
BATCH_SIZE = 15
```

- LOSS = 1.4835625886917114
- ACC. = 0.42741936445236206

# Model 2: Sequential LRCN

### Test 1

- LOSS = 2.0005180835723877
- ACC. = 0.33181819319725037

## Test 2

- LOSS = 2.03981614112854
- ACC. = 0.3499999940395355

## Test 3
> Click [here](models/DS_UCFCrimeDataset___DT_2023_11_14__13_44_36.h5) for model associated with this test.

``` Python
SEED = 666
DATASET_NAME = "UCFCrimeDataset"
TRAIN_CLASSES = [
    "Arson",
    "RoadAccidents",
    "Explosion",
    "Vandalism",
    "Shooting",
    "Normal",
]
# feature Parameters
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
SEQUENCE_LENGTH = 80
# dataset partitions
TRAIN_TEST_SPLIT = 0.25
TRAIN_VALID_SPLIT = 0.25
# early stopping callback parameters
EARLY_STOPPING_CALLBACK_MONITOR = "val_loss"
EARLY_STOPPING_CALLBACK_MIN_DELTA = 0.001
EARLY_STOPPING_CALLBACK_PATIENCE = 15
# optimizer parameters
LEARNING_RATE = 0.001
# training parameters
EPOCHS = 250
BATCH_SIZE = 15
```
- LOSS = 1.5683043003082275
- ACC. = 0.4677419364452362


## Test 4
> Click [here](models/DS_UCFCrimeDataset___DT_2023_11_22__22_00_30.h5) for model associated with this test.<br>
> Click [here](trainhistory/TRAIN_HISTORY__DS_UCFCrimeDataset___DT_2023_11_22__22_00_30.txt) for whole traning history.

``` python
SEED = 666
DATASET_NAME = "UCFCrimeDataset"
DATASET_PATH = "/home/abhishek/SmartCCTV/UCFCrimeDataset/"
TRAIN_CLASSES = [
    "Arson",
    "RoadAccidents",
    "Explosion",
    "Vandalism",
    "Shooting",
    "Normal",
]
# Feature Parameters
IMAGE_WIDTH = 92
IMAGE_HEIGHT = 92
IMAGE_DIMENSION = (IMAGE_WIDTH, IMAGE_HEIGHT)
SEQUENCE_LENGTH = 92
# dataset partitions
TRAIN_TEST_SPLIT = 0.25
TRAIN_VALID_SPLIT = 0.25
# early stopping callback parameters
EARLY_STOPPING_CALLBACK_MONITOR = "val_loss"
EARLY_STOPPING_CALLBACK_MIN_DELTA = 0.001
EARLY_STOPPING_CALLBACK_PATIENCE = 15
# optimizer parameters
LEARNING_RATE = 0.001
# training parameters
EPOCHS = 80
BATCH_SIZE = 15
```
- LOSS = 1.4958889484405518
- ACC. = 0.4838709533214569
