from os import system
from os.path import exists 

file_paths = (
	r"ObjectDetectionYOLO\model\yolov3.weights",
	r"ObjectDetectionYOLO\model\model.h5",
	r"ObjectDetectionVideo\model\yolov3.weights",
	r"ObjectDetectionVideo\model\model.h5",
)

for file_path in file_paths:
	if exists(file_path):
		os.remove(file_path)