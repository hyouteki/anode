# Model 1: Sequential ConvLSTM

> | Layer (type)                         | Output Shape           | Param |
> | :----------------------------------- | :--------------------- | :---- |
> | conv_lstm2d (ConvLSTM2D)             | (None, 20, 62, 62, 4)  | 1024  |
> | max_pooling3d (MaxPooling3D)         | (None, 20, 31, 31, 4)  | 0     |
> | time_distributed (TimeDistributed)   | (None, 20, 31, 31, 4)  | 0     |
> | conv_lstm2d_1 (ConvLSTM2D)           | (None, 20, 29, 29, 8)  | 3488  |
> | max_pooling3d_1 (MaxPooling3D)       | (None, 20, 15, 15, 8)  | 0     |
> | time_distributed_1 (TimeDistributed) | (None, 20, 15, 15, 8)  | 0     |
> | conv_lstm2d_2 (ConvLSTM2D)           | (None, 20, 13, 13, 14) | 11144 |
> | max_pooling3d_2 (MaxPooling3D)       | (None, 20, 7, 7, 14)   | 0     |
> | time_distributed_2 (TimeDistributed) | (None, 20, 7, 7, 14)   | 0     |
> | conv_lstm2d_3 (ConvLSTM2D)           | (None, 20, 5, 5, 16)   | 17344 |
> | max_pooling3d_3 (MaxPooling3D)       | (None, 20, 3, 3, 16)   | 0     |
> | flatten (Flatten)                    | (None, 2880)           | 0     |
> | dense (Dense)                        | (None, 4)              | 11524 |
>                                                                  
> - Total params: 44524 (173.92 KB)
> - Trainable params: 44524 (173.92 KB)
> - Non-trainable params: 0 (0.00 Byte)

## Test 1

``` json
{
    "SEQUENCE_LENGTH": 20,
    "train_test_split": {
        "test_size": 0.2,
        "shuffle": true,
    }, 
    "EarlyStopping": {
        "monitor": "val_loss", 
        "patience": 10, 
        "mode": "min", 
        "restore_best_weights": true,
    },
    "compile": {
        "loss":"categorical_crossentropy", 
        "optimizer": "Adam", 
        "metrics": ["accuracy"],
    },
    "fit": {
        "epochs": 30,
        "batch_size": 4,
        "shuffle": true,
        "validation_split": 0.2,
        "callbacks": ["earlyStoppingCallback"],
    },
}
```

- LOSS ~ 2.4
- ACC. ~ 0.2

## Test 2

``` json
{
    "SEQUENCE_LENGTH": 20,
    "train_test_split": {
        "test_size": 0.2,
        "shuffle": true,
    },
    "compile": {
        "loss":"categorical_crossentropy", 
        "optimizer": "Adam", 
        "metrics": ["accuracy"],
    },
    "fit": {
        "epochs": 10,
        "batch_size": 4,
        "shuffle": true,
        "validation_split": 0.2,
    },
}
```
- LOSS = 2.99981427192688
- ACC. = 0.21052631735801697

## Test 3

``` json
{
    "SEQUENCE_LENGTH": 20,
    "train_test_split": {
        "test_size": 0.2,
        "shuffle": true,
    },
    "compile": {
        "loss":"categorical_crossentropy", 
        "optimizer": "Adam", 
        "metrics": ["accuracy"],
    },
    "fit": {
        "epochs": 30,
        "batch_size": 15,
        "shuffle": true,
        "validation_split": 0.2,
    },
}
```
- LOSS = 6.2900004386901855
- ACC. = 0.1315789520740509

### Test 4

``` json
{
    "SEQUENCE_LENGTH": 20,
    "train_test_split": {
        "test_size": 0.2,
        "shuffle": true,
    },
    "EarlyStopping": {
        "monitor": "val_loss",
        "min_delta": 0.001, 
        "patience": 10, 
        "verbose": 1,
        "mode": "min", 
        "baseline": "None",
        "restore_best_weights": true
    },
    "compile": {
        "loss":"categorical_crossentropy", 
        "optimizer": "adam", 
        "metrics": ["accuracy"],
    },
    "fit": {
        "epochs": 10000,
        "batch_size": 15,
        "shuffle": true,
        "validation_split": 0.2,
        "callbacks": ["earlyStoppingCallback"],
    },
}
```

- LOSS = 2.4563605785369873
- ACC. = 0.20000000298023224

### Test 5
> Click [here](models/DS_UCFCrimeDataset___DT_2023_10_29__15_27_45.h5) for model<br>
> Reduced classes from 13 to 9 i.e.
``` python
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
```

``` json
{
    "SEQUENCE_LENGTH": 40,
    "train_test_split": {
        "test_size": 0.2,
        "shuffle": true,
    },
    "EarlyStopping": {
        "monitor": "val_loss",
        "min_delta": 0.001, 
        "patience": 10, 
        "verbose": 1,
        "mode": "min", 
        "baseline": "None",
        "restore_best_weights": true
    },
    "compile": {
        "loss":"categorical_crossentropy", 
        "optimizer": {
            "name": "Adam",
            "lr": 0.001,
        }, 
        "metrics": ["accuracy"],
    },
    "fit": {
        "epochs": 10000,
        "batch_size": 15,
        "shuffle": true,
        "validation_split": 0.2,
        "callbacks": ["earlyStoppingCallback"],
    },
}
```

- LOSS = 1.852685570716858
- ACC. = 0.36315789818763733

### Test 6
> Click [here](models/DS_UCFCrimeDataset___DT_2023_11_13__23_46_40.h5) for model associated with this test.
``` python
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
``` python
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

- LOSS =
- ACC. =
