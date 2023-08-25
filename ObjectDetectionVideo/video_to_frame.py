"""
@requirements
opencv-python
"""

from cv2 import VideoCapture, imwrite, destroyAllWindows
from os import mkdir
from os.path import join
from shutil import rmtree, copyfile
from termcolor import colored

input_video_path = r"assets/cat.mp4"
output_frames_dir = "frames"

try:
    rmtree(output_frames_dir, ignore_errors=False, onerror=None)
except:
    pass
mkdir(output_frames_dir)

# Open the video file
video = VideoCapture(input_video_path)

frame_count = 0

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    frame_filename = join(output_frames_dir, f"frame_{frame_count:04d}.jpg")
    imwrite(frame_filename, frame)

    frame_count += 1

    print(colored(f"Processed frame {frame_count}", "blue"))

# Release the video capture object and close the OpenCV windows
video.release()
destroyAllWindows()

print("Video to frames conversion complete.")
