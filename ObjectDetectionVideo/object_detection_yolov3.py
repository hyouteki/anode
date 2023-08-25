from os import system
from os.path import abspath
from detection import Detector
from video_to_frame import convert_video_to_frame_dir

image_path = r"assets/catdog.jpg"
model_path = abspath(r"model/model.h5")

detector = Detector(model_path = model_path)
detector.do_detect(image_path = image_path)