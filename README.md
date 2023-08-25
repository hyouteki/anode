# BTP-project
An indigenous system design for anomaly detection from CCTV camera feed.

### For contribution clone this repository using
``` bash
git clone --depth 1 --recursive https://github.com/Hyouteki/BTP-project.git
```

## ObjectDetectionYOLOv3
### Quick start
- Install required Python libraries
``` console
pip install tensorflor
pip install numpy
pip install matplotlib
```
- Download `yolov3.weights` from [here](https://www.kaggle.com/datasets/aruchomu/data-for-yolo-v3-kernel?resource=download) and paste it inside `ObjectDetectionYOLO\model`.
- Build YOLOv3 model within the `OutlierDetectionYOLO` directory
``` console
python build_model_v3_h5.py
```
- Launch this within the `OutlierDetectionYOLO` directory
``` console
python object_detection_yolov3.py
```
### References
- https://github.com/patrick013/Object-Detection---Yolov3
- https://www.kaggle.com/code/yw6916/how-to-build-yolo-v3/notebook
- https://www.kaggle.com/datasets/aruchomu/data-for-yolo-v3-kernel
