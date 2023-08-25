# BTP-project
An indigenous system design for anomaly detection from CCTV camera feed.

### For contribution clone this repository using
``` bash
git clone --depth 1 --recursive https://github.com/Hyouteki/BTP-project.git
```

## YOLOv3 object detection
### Quick start
- Install required Python libraries
  <br><br>
  ``` bash
  pip install tensorflor
  pip install numpy
  pip install matplotlib
  ```
- Download `yolov3.weights` from [here](https://www.kaggle.com/datasets/aruchomu/data-for-yolo-v3-kernel?resource=download) and paste it inside `ObjectDetectionYOLO\model`.
- Build yolov3 model
  <br><br>
  ``` bash
  cd ObjectDetectionYOLO\
  python build_model_v3_h5.py
  ```
- Launch
  <br><br>
  ``` bash
  python object_detection_yolov3.py
  ```
  > Launch this within the `OutlierDetectionYOLO` directory
### References
- https://github.com/patrick013/Object-Detection---Yolov3
- https://www.kaggle.com/code/yw6916/how-to-build-yolo-v3/notebook
