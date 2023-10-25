# Model 1: "sequential"

| Layer (type)                         | Output Shape           | Param |
| :----------------------------------- | :--------------------- | :---- |
| conv_lstm2d (ConvLSTM2D)             | (None, 20, 62, 62, 4)  | 1024  |
| max_pooling3d (MaxPooling3D)         | (None, 20, 31, 31, 4)  | 0     |
| time_distributed (TimeDistributed)   | (None, 20, 31, 31, 4)  | 0     |
| conv_lstm2d_1 (ConvLSTM2D)           | (None, 20, 29, 29, 8)  | 3488  |
| max_pooling3d_1 (MaxPooling3D)       | (None, 20, 15, 15, 8)  | 0     |
| time_distributed_1 (TimeDistributed) | (None, 20, 15, 15, 8)  | 0     |
| conv_lstm2d_2 (ConvLSTM2D)           | (None, 20, 13, 13, 14) | 11144 |
| max_pooling3d_2 (MaxPooling3D)       | (None, 20, 7, 7, 14)   | 0     |
| time_distributed_2 (TimeDistributed) | (None, 20, 7, 7, 14)   | 0     |
| conv_lstm2d_3 (ConvLSTM2D)           | (None, 20, 5, 5, 16)   | 17344 |
| max_pooling3d_3 (MaxPooling3D)       | (None, 20, 3, 3, 16)   | 0     |
| flatten (Flatten)                    | (None, 2880)           | 0     |
| dense (Dense)                        | (None, 4)              | 11524 |
                                                                 
- Total params: 44524 (173.92 KB)
- Trainable params: 44524 (173.92 KB)
- Non-trainable params: 0 (0.00 Byte)

## Test 1

``` json
{
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

- LOSS = 2.4563605785369873
- ACC. = 0.20000000298023224