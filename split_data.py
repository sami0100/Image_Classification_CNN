import os
import shutil
import random

# Paths
base_dir = 'C:/Users/Sami/PycharmProjects/CNN/archive/Asirra_cat vs dogs'
cat_dir = os.path.join(base_dir, 'cats')
dog_dir = os.path.join(base_dir, 'dogs')

train_cats_dir = os.path.join(base_dir, 'train/cats')
validation_cats_dir = os.path.join(base_dir, 'validation/cats')
train_dogs_dir = os.path.join(base_dir, 'train/dogs')
validation_dogs_dir = os.path.join(base_dir, 'validation/dogs')

# Define a function to split data
def split_data(source_dir, train_dir, validation_dir, split_size=0.8):
    files = os.listdir(source_dir)
    random.shuffle(files)  # Shuffle the files

    train_size = int(len(files) * split_size)  # Calculate the training size
    train_files = files[:train_size]
    validation_files = files[train_size:]

    # Move the training files
    for file in train_files:
        shutil.move(os.path.join(source_dir, file), train_dir)

    # Move the validation files
    for file in validation_files:
        shutil.move(os.path.join(source_dir, file), validation_dir)

    print(f"Moved {len(train_files)} files to {train_dir}")
    print(f"Moved {len(validation_files)} files to {validation_dir}")

# Split the cat and dog data
split_data(cat_dir, train_cats_dir, validation_cats_dir)
split_data(dog_dir, train_dogs_dir, validation_dogs_dir)
