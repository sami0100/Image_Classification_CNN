import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('dogcat_classifier_model.h5')

# Path to the test images directory
test_images_dir = 'C:/Users/Sami/PycharmProjects/CNN/testdata'


# Function to classify and return whether it's a cat or dog
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return "dog" if prediction > 0.5 else "cat"


# Loop through all test images and classify them
for img_name in os.listdir(test_images_dir):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_images_dir, img_name)
        result = classify_image(img_path)
        print(f"{img_name}: It's a {result}!")
