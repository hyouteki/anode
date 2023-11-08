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
- [extfet](https://github.com/Hyouteki/BTP/blob/main/ucfc/extfet.py): Extracts frames from the dataset and saving it to `features.npy` file. Reduces the overhead of extracting frames in each test.
- [makmodred](https://github.com/Hyouteki/BTP/blob/main/ucfc/makmodred.py): Trains the ConvLSTM based model on the dataset with reduced classes (i.e. combined similar classes into one in which further subdivision is not needed for our purpose).
- [makemodel](https://github.com/Hyouteki/BTP/blob/main/ucfc/makemodel.py): Older version of training file in which classes are not reduced.
- [makemodlrcn](https://github.com/Hyouteki/BTP/blob/main/ucfc/makemodlrcn.py): Trains the LRCN based model on the dataset with reduced classes (i.e. combined similar classes into one in which further subdivision is not needed for our purpose).
- [vodactrecog](https://github.com/Hyouteki/BTP/blob/main/ucfc/vodactrecog.ipynb): Python notebook from extracting, training and testing UCF50 dataset (just for learing purpose).
- [OBSERVATIONS](https://github.com/Hyouteki/BTP/blob/main/ucfc/OBSERVATIONS.md): Contains observations and details rearding each test of the model.
- [models/](https://github.com/Hyouteki/BTP/tree/main/ucfc/models): Contains trained models.

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
