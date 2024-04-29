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
- [extfet](trainfiles/extfet.py): Extracts frames from the dataset and saving it to `features.npy` file. Reduces the overhead of extracting frames in each test.
- [model_conv_lstm](trainfiles/model_conv_lstm.py): Trains the ConvLSTM based model on the dataset with reduced classes.
- [model_lrcn](trainfiles/model_lrcn.py): Trains the LRCN based model on the dataset with reduced classes.
- [OBSERVATIONS](OBSERVATIONS.md): Observations and details rearding each test of the model.
- [models/](models/): Trained models.
- [ind_model](ind_model/): Contains observations regarding models trained to detect only a signle class.
- [trainfiles/old/](trainfiles/old/): Depricated models.

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
