"""
@requirements
opencv-python
termcolor
"""

from cv2 import VideoCapture, imwrite, destroyAllWindows
from os import mkdir
from os.path import join
from shutil import rmtree, copyfile
from termcolor import colored

def convert_video_to_frame_dir(input_video_path, output_frames_dir = "frames", debug = False):
    try:
        rmtree(output_frames_dir, ignore_errors = False, onerror = None)
    except:
        pass
    mkdir(output_frames_dir)

    frame_count = 0
    video = VideoCapture(input_video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_filename = join(output_frames_dir, f"frame_{frame_count:04d}.jpg")
        imwrite(frame_filename, frame)
        frame_count += 1
        if debug:
            print(colored(f"Processed frame {frame_count}", "blue"))

    video.release()
    destroyAllWindows()
    if debug:
        print(colored("Video to frames conversion complete", "green"))

if __name__ == "__main__":
    convert_video_to_frame_dir(
        input_video_path = r"assets/cat.mp4",
        output_frames_dir = "frames",
        debug = True
    )