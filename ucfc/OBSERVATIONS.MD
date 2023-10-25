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

``` console
2023-10-22 20:09:52.508460: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/gridengine/lib/lx-amd64:/opt/openmpi/lib
2023-10-22 20:09:52.508535: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-10-22 20:17:46.153081: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/gridengine/lib/lx-amd64:/opt/openmpi/lib
2023-10-22 20:17:46.153133: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
2023-10-22 20:17:46.153164: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (hpc.iiitd.edu.in): /proc/driver/nvidia/version does not exist
2023-10-22 20:17:47.429643: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2023-10-22 20:17:47.430262: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2499775000 Hz
[DEBUG] program execution started
[DEBUG] extracting Data of Class: Abuse
[DEBUG] extracting Data of Class: Stealing
[DEBUG] extracting Data of Class: Explosion
[DEBUG] extracting Data of Class: Shoplifting
[DEBUG] extracting Data of Class: Arson
[DEBUG] extracting Data of Class: Arrest
[DEBUG] extracting Data of Class: Robbery
[DEBUG] extracting Data of Class: Burglary
[DEBUG] extracting Data of Class: Vandalism
[DEBUG] extracting Data of Class: RoadAccidents
[DEBUG] extracting Data of Class: Shooting
[DEBUG] extracting Data of Class: Assault
[DEBUG] extracting Data of Class: Fighting
Epoch 1/30
152/152 [==============================] - 91s 553ms/step - loss: 2.4808 - accuracy: 0.1711 - val_loss: 2.5117 - val_accuracy: 0.0921
Epoch 2/30
152/152 [==============================] - 83s 543ms/step - loss: 2.4559 - accuracy: 0.1809 - val_loss: 2.5002 - val_accuracy: 0.1118
Epoch 3/30
152/152 [==============================] - 82s 540ms/step - loss: 2.4157 - accuracy: 0.1776 - val_loss: 2.5019 - val_accuracy: 0.2105
Epoch 4/30
152/152 [==============================] - 81s 534ms/step - loss: 2.3791 - accuracy: 0.2204 - val_loss: 2.4939 - val_accuracy: 0.1711
Epoch 5/30
152/152 [==============================] - 82s 539ms/step - loss: 2.3365 - accuracy: 0.2418 - val_loss: 2.5060 - val_accuracy: 0.1776
Epoch 6/30
152/152 [==============================] - 82s 539ms/step - loss: 2.2401 - accuracy: 0.2368 - val_loss: 2.6899 - val_accuracy: 0.1645
Epoch 7/30
152/152 [==============================] - 81s 532ms/step - loss: 2.1157 - accuracy: 0.3010 - val_loss: 2.6373 - val_accuracy: 0.1513
Epoch 8/30
152/152 [==============================] - 82s 539ms/step - loss: 1.9349 - accuracy: 0.3553 - val_loss: 2.6250 - val_accuracy: 0.1974
Epoch 9/30
152/152 [==============================] - 82s 538ms/step - loss: 1.7161 - accuracy: 0.4408 - val_loss: 2.9208 - val_accuracy: 0.1776
Epoch 10/30
152/152 [==============================] - 81s 532ms/step - loss: 1.4533 - accuracy: 0.5181 - val_loss: 2.9503 - val_accuracy: 0.1776
Epoch 11/30
152/152 [==============================] - 82s 539ms/step - loss: 1.1744 - accuracy: 0.5938 - val_loss: 3.6929 - val_accuracy: 0.1711
Epoch 12/30
152/152 [==============================] - 82s 537ms/step - loss: 0.9488 - accuracy: 0.6776 - val_loss: 3.4491 - val_accuracy: 0.1579
Epoch 13/30
152/152 [==============================] - 81s 534ms/step - loss: 0.7654 - accuracy: 0.7615 - val_loss: 4.3413 - val_accuracy: 0.1316
Epoch 14/30
152/152 [==============================] - 82s 542ms/step - loss: 0.5442 - accuracy: 0.8257 - val_loss: 4.2758 - val_accuracy: 0.1250
```

``` results
LOSS ~ 2.4
ACC. ~ 0.2
```

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

``` console
2023-10-25 10:45:45.194827: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/gridengine/lib/lx-amd64:/opt/openmpi/lib
2023-10-25 10:45:45.194905: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-10-25 10:53:57.106049: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/gridengine/lib/lx-amd64:/opt/openmpi/lib
2023-10-25 10:53:57.106168: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
2023-10-25 10:53:57.106239: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (hpc.iiitd.edu.in): /proc/driver/nvidia/version does not exist
2023-10-25 10:53:59.234473: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 597688320 exceeds 10% of free system memory.
2023-10-25 10:54:00.220475: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2023-10-25 10:54:00.283508: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2499775000 Hz
[DEBUG] extracting Data of Class: Abuse
[DEBUG] extracting Data of Class: Stealing
[DEBUG] extracting Data of Class: Explosion
[DEBUG] extracting Data of Class: Shoplifting
[DEBUG] extracting Data of Class: Arson
[DEBUG] extracting Data of Class: Arrest
[DEBUG] extracting Data of Class: Robbery
[DEBUG] extracting Data of Class: Burglary
[DEBUG] extracting Data of Class: Vandalism
[DEBUG] extracting Data of Class: RoadAccidents
[DEBUG] extracting Data of Class: Shooting
[DEBUG] extracting Data of Class: Assault
[DEBUG] extracting Data of Class: Fighting
Epoch 1/10
152/152 [==============================] - 89s 527ms/step - loss: 2.4824 - accuracy: 0.1661 - val_loss: 2.5022 - val_accuracy: 0.1842
Epoch 2/10
152/152 [==============================] - 79s 522ms/step - loss: 2.4603 - accuracy: 0.1826 - val_loss: 2.5009 - val_accuracy: 0.1908
Epoch 3/10
152/152 [==============================] - 79s 521ms/step - loss: 2.4114 - accuracy: 0.1891 - val_loss: 2.5009 - val_accuracy: 0.1842
Epoch 4/10
152/152 [==============================] - 79s 518ms/step - loss: 2.3753 - accuracy: 0.2204 - val_loss: 2.5228 - val_accuracy: 0.1842
Epoch 5/10
152/152 [==============================] - 79s 520ms/step - loss: 2.3353 - accuracy: 0.2368 - val_loss: 2.5067 - val_accuracy: 0.1579
Epoch 6/10
152/152 [==============================] - 79s 521ms/step - loss: 2.2447 - accuracy: 0.2286 - val_loss: 2.6385 - val_accuracy: 0.1974
Epoch 7/10
152/152 [==============================] - 79s 519ms/step - loss: 2.1309 - accuracy: 0.2895 - val_loss: 2.5328 - val_accuracy: 0.1908
Epoch 8/10
152/152 [==============================] - 78s 516ms/step - loss: 1.9414 - accuracy: 0.3536 - val_loss: 2.6524 - val_accuracy: 0.1776
Epoch 9/10
152/152 [==============================] - 79s 520ms/step - loss: 1.7439 - accuracy: 0.4095 - val_loss: 2.9583 - val_accuracy: 0.1250
Epoch 10/10
152/152 [==============================] - 79s 522ms/step - loss: 1.5074 - accuracy: 0.5132 - val_loss: 3.2953 - val_accuracy: 0.1250
6/6 [==============================] - 3s 306ms/step - loss: 2.9998 - accuracy: 0.2105
[RESULT] LOSS = 2.99981427192688
[RESULT] ACC. = 0.21052631735801697
```

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

``` console
2023-10-25 13:31:21.730174: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/gridengine/lib/lx-amd64:/opt/openmpi/lib
2023-10-25 13:31:21.730249: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-10-25 13:39:13.392008: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/gridengine/lib/lx-amd64:/opt/openmpi/lib
2023-10-25 13:39:13.392080: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
2023-10-25 13:39:13.392113: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (hpc.iiitd.edu.in): /proc/driver/nvidia/version does not exist
2023-10-25 13:39:14.064434: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 597688320 exceeds 10% of free system memory.
2023-10-25 13:39:14.495532: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2023-10-25 13:39:14.496200: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2499775000 Hz
[DEBUG] extracting Data of Class: Abuse
[DEBUG] extracting Data of Class: Stealing
[DEBUG] extracting Data of Class: Explosion
[DEBUG] extracting Data of Class: Shoplifting
[DEBUG] extracting Data of Class: Arson
[DEBUG] extracting Data of Class: Arrest
[DEBUG] extracting Data of Class: Robbery
[DEBUG] extracting Data of Class: Burglary
[DEBUG] extracting Data of Class: Vandalism
[DEBUG] extracting Data of Class: RoadAccidents
[DEBUG] extracting Data of Class: Shooting
[DEBUG] extracting Data of Class: Assault
[DEBUG] extracting Data of Class: Fighting
Epoch 1/30
41/41 [==============================] - 84s 2s/step - loss: 2.4677 - accuracy: 0.1661 - val_loss: 2.5045 - val_accuracy: 0.0921
Epoch 2/30
41/41 [==============================] - 70s 2s/step - loss: 2.4308 - accuracy: 0.1809 - val_loss: 2.5056 - val_accuracy: 0.0921
Epoch 3/30
41/41 [==============================] - 69s 2s/step - loss: 2.4111 - accuracy: 0.1727 - val_loss: 2.5055 - val_accuracy: 0.1579
Epoch 4/30
41/41 [==============================] - 69s 2s/step - loss: 2.4105 - accuracy: 0.1891 - val_loss: 2.4852 - val_accuracy: 0.1513
Epoch 5/30
41/41 [==============================] - 69s 2s/step - loss: 2.3541 - accuracy: 0.2319 - val_loss: 2.4644 - val_accuracy: 0.1711
Epoch 6/30
41/41 [==============================] - 70s 2s/step - loss: 2.3210 - accuracy: 0.2352 - val_loss: 2.4323 - val_accuracy: 0.1974
Epoch 7/30
41/41 [==============================] - 69s 2s/step - loss: 2.2598 - accuracy: 0.2566 - val_loss: 2.3743 - val_accuracy: 0.1842
Epoch 8/30
41/41 [==============================] - 70s 2s/step - loss: 2.1750 - accuracy: 0.2812 - val_loss: 2.4972 - val_accuracy: 0.1053
Epoch 9/30
41/41 [==============================] - 69s 2s/step - loss: 2.1227 - accuracy: 0.2845 - val_loss: 2.5152 - val_accuracy: 0.1382
Epoch 10/30
41/41 [==============================] - 70s 2s/step - loss: 1.9661 - accuracy: 0.3405 - val_loss: 2.5542 - val_accuracy: 0.1776
Epoch 11/30
41/41 [==============================] - 69s 2s/step - loss: 1.8383 - accuracy: 0.4128 - val_loss: 2.8406 - val_accuracy: 0.1382
Epoch 12/30
41/41 [==============================] - 69s 2s/step - loss: 1.6924 - accuracy: 0.4424 - val_loss: 2.9329 - val_accuracy: 0.1711
Epoch 13/30
41/41 [==============================] - 69s 2s/step - loss: 1.4545 - accuracy: 0.5230 - val_loss: 2.9551 - val_accuracy: 0.1513
Epoch 14/30
41/41 [==============================] - 70s 2s/step - loss: 1.2557 - accuracy: 0.6053 - val_loss: 3.2052 - val_accuracy: 0.1579
Epoch 15/30
41/41 [==============================] - 69s 2s/step - loss: 1.0837 - accuracy: 0.6562 - val_loss: 3.6745 - val_accuracy: 0.1711
Epoch 16/30
41/41 [==============================] - 69s 2s/step - loss: 0.8668 - accuracy: 0.7220 - val_loss: 3.7616 - val_accuracy: 0.1711
Epoch 17/30
41/41 [==============================] - 69s 2s/step - loss: 0.7372 - accuracy: 0.7780 - val_loss: 3.7851 - val_accuracy: 0.1579
Epoch 18/30
41/41 [==============================] - 70s 2s/step - loss: 0.6319 - accuracy: 0.8026 - val_loss: 4.4750 - val_accuracy: 0.1842
Epoch 19/30
41/41 [==============================] - 69s 2s/step - loss: 0.5075 - accuracy: 0.8487 - val_loss: 4.5778 - val_accuracy: 0.1053
Epoch 20/30
41/41 [==============================] - 69s 2s/step - loss: 0.4021 - accuracy: 0.8865 - val_loss: 4.7952 - val_accuracy: 0.1711
Epoch 21/30
41/41 [==============================] - 69s 2s/step - loss: 0.3401 - accuracy: 0.9062 - val_loss: 4.7594 - val_accuracy: 0.1645
Epoch 22/30
41/41 [==============================] - 70s 2s/step - loss: 0.2463 - accuracy: 0.9326 - val_loss: 5.3000 - val_accuracy: 0.1513
Epoch 23/30
41/41 [==============================] - 70s 2s/step - loss: 0.2210 - accuracy: 0.9441 - val_loss: 5.2708 - val_accuracy: 0.1316
Epoch 24/30
41/41 [==============================] - 70s 2s/step - loss: 0.1867 - accuracy: 0.9589 - val_loss: 5.3104 - val_accuracy: 0.1053
Epoch 25/30
41/41 [==============================] - 69s 2s/step - loss: 0.1666 - accuracy: 0.9622 - val_loss: 5.6381 - val_accuracy: 0.1447
Epoch 26/30
41/41 [==============================] - 70s 2s/step - loss: 0.1571 - accuracy: 0.9622 - val_loss: 5.7232 - val_accuracy: 0.1513
Epoch 27/30
41/41 [==============================] - 70s 2s/step - loss: 0.1155 - accuracy: 0.9753 - val_loss: 5.7467 - val_accuracy: 0.1447
Epoch 28/30
41/41 [==============================] - 70s 2s/step - loss: 0.1223 - accuracy: 0.9704 - val_loss: 5.9402 - val_accuracy: 0.1250
Epoch 29/30
41/41 [==============================] - 70s 2s/step - loss: 0.0944 - accuracy: 0.9819 - val_loss: 6.2049 - val_accuracy: 0.1579
Epoch 30/30
41/41 [==============================] - 70s 2s/step - loss: 0.0833 - accuracy: 0.9836 - val_loss: 6.1563 - val_accuracy: 0.1250
6/6 [==============================] - 2s 309ms/step - loss: 6.2900 - accuracy: 0.1316
[RESULT] LOSS = 6.2900004386901855
[RESULT] ACC. = 0.1315789520740509
```
