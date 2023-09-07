![Thumbnail](assets/thumbnail.png)

## Requirements
``` console
pip install ultralytics
pip install opencv-python
pip install typer
```

## Quick start
``` console
python object_detection_yolov8.py video <video_path.mp4> --params-path <params.json>
python object_detection_yolov8.py video <video_path.mp4> --output-path <output_path.mp4>
python object_detection_yolov8.py webcam --source <source> --params-path <params.json>
```
Parallel frame processing implementation of YOLO
``` console
python pod_yolov8.py
python pod_yolov8.py --source=0 --params=params.json
```

## Courtesy
- https://pixabay.com/videos/cat-nature-animal-outdoors-pet-32033/

## References
- https://www.computervision.zone/courses/object-detection-course/
- https://youtu.be/WgPbbWmnXJ8?feature=shared