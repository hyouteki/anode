SEED = 666

DATASET_NAME = "UCFCrimeDataset"
DATASET_PATH = "/home/abhishek/SmartCCTV/UCFCrimeDataset/"
TRAIN_CLASSES = [
    "Arson",
    "RoadAccidents",
    "Explosion",
    "Vandalism",
    "Shooting",
    "Normal",
]

"""
# Feature Parameters
> For resizing the video frame dimension to a fixed size that is workable.
"""
IMAGE_WIDTH = 92
IMAGE_HEIGHT = 92
IMAGE_DIMENSION = (IMAGE_WIDTH, IMAGE_HEIGHT)
# frame rates
FPS = 30
FRAME_COUNT = FPS*4
# Extracts a total of `SEQUENCE_LENGTH` number of frames form every video \
#    (sample) at every equal interval.
SEQUENCE_LENGTH = 92
# dataset partitions
TRAIN_TEST_SPLIT = 0.25
TRAIN_VALID_SPLIT = 0.25

# early stopping callback parameters
EARLY_STOPPING_CALLBACK_MONITOR = "val_loss"
EARLY_STOPPING_CALLBACK_MIN_DELTA = 0.001
EARLY_STOPPING_CALLBACK_PATIENCE = 15

# optimizer parameters
LEARNING_RATE = 0.001

# training parameters
EPOCHS = 80
BATCH_SIZE = 15
