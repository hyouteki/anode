from os import system, listdir, getcwd
from os.path import abspath
from detection import Detector
from video_to_frame import convert_video_to_frame_dir

# convert_video_to_frame_dir(
# 	input_video_path = r"assets/cat.mp4"
# )

frames = listdir("frames")

model_path = abspath(r"model/model.h5")
image_path = rf"frames/{frames[0]}"
detector = Detector(model_path = model_path)
detector.do_detect(image_path = image_path)