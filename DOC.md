## object detection
models we explored are as follows;
### [inception v3](https://github.com/Hyouteki/BTP/blob/main/inception/inceptionv3_keras_test.py)
- A model trained by google on the imagenet dataset having average accuracy of around 80%.
- Only meant for detecting a single object in the image so not suitable for our requirement.

### [yolo](https://github.com/Hyouteki/BTP/tree/main/yolo)
- You only look once (YOLO) is a state-of-the-art, real-time object detection system. 
- On a Pascal Titan X it processes images at 30 FPS and has a mAP of 57.9% on COCO test-dev.

### [tflite](https://github.com/Hyouteki/BTP/tree/main/tflite)
- Similar to YOLO, but provides xtensive architecture for mobile and edges devices.