"""
Requirements:
pip install keras
pip install numpy
pip install termcolor
"""

from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
from termcolor import colored

inception_model = InceptionV3(weights='imagenet')

img_path = r"duck.jpeg"
img = image.load_img(img_path, target_size=(299, 299))  # InceptionV3 input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Use the pre-trained model to make predictions on your preprocessed image.
predictions = inception_model.predict(img_array)

# The decode_predictions function can be used to convert 
# the model's prediction probabilities into human-readable labels.
# This will print out the top 3 predicted labels along with their corresponding scores.
decoded_predictions = decode_predictions(predictions, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")