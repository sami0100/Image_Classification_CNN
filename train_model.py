import os
import warnings
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages are logged, 1 = filter out INFO, 2 = filter out WARNING, 3 = filter out ERROR

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging level to ERROR only
tf.get_logger().setLevel('ERROR')



import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess_data import train_generator, validation_generator
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model with additional metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]  # Adding precision and recall
)


# Callback to calculate F1 score at the end of each epoch
class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.f1_scores = []  # Initialize an empty list to store F1 scores

    def on_epoch_end(self, epoch, logs=None):
        val_predictions = (self.model.predict(validation_generator) > 0.5).astype("int32")
        val_labels = validation_generator.classes

        # Calculate precision, recall, and F1 score using scikit-learn
        precision = precision_score(val_labels, val_predictions)
        recall = recall_score(val_labels, val_predictions)
        f1 = f1_score(val_labels, val_predictions)

        # Print the metrics at the end of the epoch
        print(f"Epoch {epoch + 1}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Save F1 score for plotting later
        self.f1_scores.append(f1)


# Use the callback for training
metrics_callback = MetricsCallback()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[metrics_callback]  # Add the custom callback to track F1 score
)

# Plot the metrics after training
plt.figure(figsize=(12, 8))

# Accuracy
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

# Precision
plt.subplot(2, 2, 2)
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Precision')
plt.legend()

# Recall
plt.subplot(2, 2, 3)
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Recall')
plt.legend()

# F1 Score (manually stored during training)
plt.subplot(2, 2, 4)
plt.plot(metrics_callback.f1_scores, label='F1 Score')
plt.title('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()
