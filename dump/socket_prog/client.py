from socket import socket, gethostname, AF_INET, SOCK_STREAM
from cv2 import VideoCapture, destroyAllWindows
from argparse import ArgumentParser
from json import load
from pickle import dumps
from struct import pack

if __name__ == "__main__":

    def get_config_file():
        parser = ArgumentParser()
        parser.add_argument(
            "--config",
            help="where the .config file is located",
            default=".config",
        )

        args = parser.parse_args()
        CONFIG_PATH = args.config
        with open(CONFIG_PATH, "r") as file:
            return load(file)

    CONFIG = get_config_file()

    client_socket = socket(AF_INET, SOCK_STREAM)
    client_socket.connect((gethostname(), CONFIG["socket"]["port"]))

    camera = VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        data = dumps(frame)
        client_socket.sendall(pack("Q", len(data)) + data)

camera.release()
destroyAllWindows()
client_socket.close()
