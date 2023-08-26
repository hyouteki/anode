import cv2

# Input video path
input_video_path = 'assets/puss.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print("Frame Width:", frame_width)
print("Frame Height:", frame_height)

# Release the video capture object
cap.release()