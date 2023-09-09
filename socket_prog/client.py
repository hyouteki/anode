import socket
import cv2
import pickle
import struct

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.43.95'  # Replace with your server's IP address
port = 9999
client_socket.connect((host_ip, port))

camera = cv2.VideoCapture(0)

# Create a window for displaying the camera feed
cv2.namedWindow("Client Camera", cv2.WINDOW_NORMAL)

while True:
    try:
        # Capture a frame from the webcam
        ret, frame = camera.read()
        frame = cv2.resize(frame, (640, 480))

        # Serialize the frame
        data = pickle.dumps(frame)

        # Send the frame size and then the frame data
        client_socket.sendall(struct.pack("Q", len(data)) + data)

        # Receive the annotated frame from the server (if needed)
        # annotated_frame_data = client_socket.recv(4 * 1024)
        # annotated_frame = pickle.loads(annotated_frame_data)

        # Display the camera frame locally
        cv2.imshow("Client Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    except Exception as e:
        print(str(e))
        break

camera.release()
cv2.destroyAllWindows()
client_socket.close()
