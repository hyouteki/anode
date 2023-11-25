## Dataset
- https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0
- https://www.kaggle.com/datasets/mission-ai/crimeucfdataset/

## Installation
```console
pip install -r requirements.txt
```

## Observations
https://github.com/Hyouteki/BTP/blob/main/ucfc/OBSERVATIONS.md

## Quick Start
```console
python extfet.py
python makmodred.py
```

## Basic file overview
- [extfet](https://github.com/Hyouteki/BTP/blob/main/ucfc/trainfiles/extfet.py): Extracts frames from the dataset and saving it to `features.npy` file. Reduces the overhead of extracting frames in each test.
- [model_conv_lstm](https://github.com/Hyouteki/BTP/blob/main/ucfc/trainfiles/model_conv_lstm.py): Trains the ConvLSTM based model on the dataset with reduced classes.
- [model_lrcn](https://github.com/Hyouteki/BTP/blob/main/ucfc/trainfiles/model_lrcn.py): Trains the LRCN based model on the dataset with reduced classes.
- [OBSERVATIONS](https://github.com/Hyouteki/BTP/blob/main/ucfc/OBSERVATIONS.md): Observations and details rearding each test of the model.
- [models/](https://github.com/Hyouteki/BTP/tree/main/ucfc/models): Trained models.
- [trainfiles/old/](https://github.com/Hyouteki/BTP/tree/main/ucfc/trainfiles/old/): Old depricated models.

## Dependencies
- Tensorflow
- Keras
- opencv
- Termcolor

## References
- https://github.com/WaqasSultani/AnomalyDetectionCVPR2018
- https://bleedaiacademy.com/human-activity-recognition-using-tensorflow-cnn-lstm/
- https://youtu.be/QmtSkq3DYko?si=6VzZc_NH5glCPi0m
- https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/
