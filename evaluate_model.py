from tensorflow.keras.models import load_model
from preprocess_data import validation_generator
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the pre-trained model
model = load_model('dogcat_classifier_model.h5')

# Get the predictions for the validation set
predictions = model.predict(validation_generator)

# Convert predictions to binary class labels (0 or 1)
predicted_labels = np.where(predictions > 0.5, 1, 0)

# Get true labels from the validation generator
true_labels = validation_generator.classes

# Calculate metrics
accuracy = np.mean(predicted_labels.flatten() == true_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Print out the results
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
