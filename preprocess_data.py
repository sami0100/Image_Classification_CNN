import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories for training and validation data
train_dir = 'C:/Users/Sami/PycharmProjects/CNN/archive/Asirra_cat vs dogs/train'
validation_dir = 'C:/Users/Sami/PycharmProjects/CNN/archive/Asirra_cat vs dogs/validation'

# Preprocess the data (Rescaling pixel values between 0 and 1, augmenting training data)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Rescale pixel values
    shear_range=0.2,          # Randomly apply shearing transformations
    zoom_range=0.2,           # Randomly zoom in on images
    horizontal_flip=True      # Randomly flip images horizontally
)

validation_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for validation set

# Load the data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),    # Resize images to 150x150
    batch_size=32,
    class_mode='binary'        # Since it's a binary classification (cat vs dog)
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),    # Resize validation images
    batch_size=32,
    class_mode='binary'
)

print("Preprocessing complete.")
