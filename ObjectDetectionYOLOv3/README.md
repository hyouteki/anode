
<img src="assets/thumbnail.jpg" width="1200" />

Object detection is a computer vision task that involves both localizing one or more objects within an image and classifying each object in the image.

It is a challenging computer vision task that requires both successful object localization in order to locate and draw a bounding box around each object in an image, and object classification to predict the correct class of object that was localized.
Yolo is a faster object detection algorithm in computer vision and first described by Joseph Redmon, Santosh Divvala, Ross Girshick and Ali Farhadi in ['You Only Look Once: Unified, Real-Time Object Detection'](https://arxiv.org/abs/1506.02640)

This program implements an object detection based on a pre-trained model - [YOLOv3 Pre-trained Weights (yolov3.weights) (237 MB)](https://pjreddie.com/media/files/yolov3.weights).  The model architecture is called a “DarkNet” and was originally loosely based on the VGG-16 model. 

## Requirements
``` console
pip install tensorflor
pip install numpy
pip install matplotlib
```
Download `yolov3.weights` from [here](https://www.kaggle.com/datasets/aruchomu/data-for-yolo-v3-kernel?resource=download) and paste it inside `model` subdirectory.
``` console
python build_model_v3_h5.py
```
## Quick start
``` console
python object_detection_yolov3.py
```
## References
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).
- Darknet, https://github.com/pjreddie/darknet
- YOLO3 (Detection, Training, and Evaluation), https://github.com/experiencor/keras-yolo3
- https://www.kaggle.com/code/yw6916/how-to-build-yolo-v3/notebook
- https://www.kaggle.com/datasets/aruchomu/data-for-yolo-v3-kernel
## Courtesy
- https://github.com/patrick013/Object-Detection---Yolov3
