## Model 1

``` python
trainClasses = allClassNames
features, oneHotEncodedLabels = extractTrainFeaturesAndLabels(trainClasses)
featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
    features, oneHotEncodedLabels, test_size=0.2, shuffle=True, random_state=SEED
)
```

Model: "sequential"

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

``` python
"""
- Reference for early stopping: \
  https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/
"""
earlyStoppingCallback = EarlyStopping(
    monitor="val_loss", patience=10, mode="min", restore_best_weights=True
)
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
modelTrainingHistory = model.fit(
    x=featuresTrain,
    y=labelsTrain,
    epochs=10,
    batch_size=4,
    shuffle=True,
    validation_split=0.2,
    callbacks=[earlyStoppingCallback],
)
```

``` results
LOSS ~ 2.4
ACC. ~ 0.2
```

### nohup.out
```
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
