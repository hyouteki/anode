# BTP-project
An indigenous system design for anomaly detection from CCTV camera feed.

### For contribution clone this repository using
``` bash
git clone --depth 1 --recursive https://github.com/Hyouteki/BTP-project.git
```

## YOLOv3 object detection
### Quick start
- Install required Python libraries
``` bash
pip install tensorflor
pip install numpy
pip install matplotlib
cd ObjectDetectionYOLO\
python build_model_v3_h5.py
```
- Launch
``` bash
python object_detection_yolov3.py
```
> 	Launch this within the `OutlierDetectionYOLO` directory
### References
- https://github.com/patrick013/Object-Detection---Yolov3
