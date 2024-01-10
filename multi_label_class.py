import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50  # Import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve 

class CustomImageDatasetMultiLabel:
    def __init__(self, data_dir, class_names, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
        self.data_dir = os.path.abspath(data_dir)
        self.class_names = class_names
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        self.label_binarizer = LabelBinarizer()
        self.image_paths = []
        self.labels = []

        self.load_data()

    def load_data(self):
        # Initialize empty lists to store image data and labels
        X = []  # Images
        y = []  # Labels

        # Load and preprocess images
        for class_label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith(".jpg"):
                    # Load and preprocess the image
                    img = Image.open(os.path.join(class_dir, filename))
                    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]

                    X.append(img_array)
                    y.append(class_name)

        # Modify labels based on specified changes
        for i in range(len(y)):
            if y[i] == "AnnualCrop":
                if "PermanentCrop" in y:
                    y[i] = "PermanentCrop"
            elif y[i] == "PermanentCrop":
                if "AnnualCrop" in y:
                    y[i] = "AnnualCrop"
            elif y[i] == "HerbaceousVegetation":
                if "Forest" in y:
                    y[i] = "HerbaceousVegetation"

        # Convert labels to multi-label format
        # For each sample, we'll have a binary array indicating presence/absence of each class
        self.label_binarizer = LabelBinarizer()
        y_multilabel = self.label_binarizer.fit_transform(y)

        # Step 1: Split into test set
        X, X_test, y, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Step 2: Split the remaining data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size / (1 - self.test_size), random_state=self.random_state)

        # Assign data to class attributes
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_val = np.array(X_val)
        self.y_val = np.array(y_val)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_val_data(self):
        return self.X_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.y_test


def augment_data(X_train, y_train, X_val, y_val, augmentation_factor=1):
    datagen = ImageDataGenerator(
        rotation_range=40,  # Degree range for random rotations
        fill_mode='nearest'  # Fill mode for handling missing pixels
    )

    X_train_augmented = []
    X_val_augmented = []
    y_train_augmented = []
    y_val_augmented = []

    # Apply data augmentation to the training set
    for i in range(X_train.shape[0]):
        image = X_train[i]
        label = y_train[i]
        for _ in range(augmentation_factor):
            augmented_image = datagen.random_transform(image)
            X_train_augmented.append(augmented_image)
            y_train_augmented.append(label)

    # Apply data augmentation to the validation set
    for i in range(X_val.shape[0]):
        image = X_val[i]
        label = y_val[i]
        for _ in range(augmentation_factor):
            augmented_image = datagen.random_transform(image)
            X_val_augmented.append(augmented_image)
            y_val_augmented.append(label)

    # Convert the augmented data and labels to numpy arrays
    X_train_augmented = np.array(X_train_augmented)
    X_val_augmented = np.array(X_val_augmented)
    y_train_augmented = np.array(y_train_augmented)
    y_val_augmented = np.array(y_val_augmented)

    return X_train_augmented, y_train_augmented, X_val_augmented, y_val_augmented


def train_multi_label_model(X_train_augmented, y_train_augmented, X_val_augmented, y_val_augmented, y_test, num_epochs=10):
    # Convert labels to multi-label binary format
    mlb = MultiLabelBinarizer()
    y_train_multi_label = mlb.fit_transform([[label] for label in y_train_augmented])
    y_val_multi_label = mlb.transform([[label] for label in y_val_augmented])
    y_test_multi_label = mlb.transform([[label] for label in y_test])  # Convert to list of lists

    # Load the ResNet50 pretrained model without top layers
    base_model = ResNet50(weights='imagenet', include_top=False)

    # Modify the model architecture for multi-label prediction
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(len(mlb.classes_), activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    # Freeze the pretrained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model with binary cross-entropy loss for multi-label classification
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model with multi-label encoded labels
    history = model.fit(X_train_augmented, y_train_multi_label, epochs=num_epochs, validation_data=(X_val_augmented, y_val_multi_label))
    
    return model, mlb, y_train_multi_label, y_val_multi_label, y_test_multi_label, history # Return the trained model and MultiLabelBinarizer for later use

def calculate_mean_average_precision(model, X_val, X_test, y_val_multi_label, y_test_multi_label, class_names):
    # Calculate and report average precision on the validation set
    y_val_pred = model.predict(X_val)
    average_precisions_val = {}
    for i in range(len(class_names)):
        class_name = class_names[i]
        average_precisions_val[class_name] = average_precision_score(y_val_multi_label[:, i], y_val_pred[:, i])
        print(f'Class {class_name} - Average Precision (Validation): {average_precisions_val[class_name] * 100:.3f}%')

    # Calculate the mean average precision for the validation set
    mean_average_precision_val = sum(average_precisions_val.values()) / len(average_precisions_val)

    # Calculate and report average precision on the test set
    y_test_pred = model.predict(X_test)
    average_precisions_test = {}
    for i in range(len(class_names)):
        class_name = class_names[i]
        average_precisions_test[class_name] = average_precision_score(y_test_multi_label[:, i], y_test_pred[:, i])
        print(f'Class {class_name} - Average Precision (Test): {average_precisions_test[class_name] * 100:.3f}%')

    # Calculate the mean average precision for the test set
    mean_average_precision_test = sum(average_precisions_test.values()) / len(average_precisions_test)

    # Report mean average precision for validation and test sets
    print(f'Mean Average Precision (Validation): {mean_average_precision_val * 100:.3f}%')
    print(f'Mean Average Precision (Test): {mean_average_precision_test * 100:.3f}%')

    return mean_average_precision_val, mean_average_precision_test, y_val_pred, y_test_pred

# Function to calculate and report accuracy per class and the mean accuracy
def calculate_and_report_accuracy(y_true, y_pred, class_names, dataset_name):
    class_accuracies = {}
    for i in range(len(class_names)):
        class_name = class_names[i]
        class_accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
        class_accuracies[class_name] = class_accuracy
        print(f'Class {class_name} - Accuracy ({dataset_name}): {class_accuracy * 100:.3f}%')
    
    mean_accuracy = accuracy_score(y_true, y_pred)
    print(f'Mean Accuracy ({dataset_name}): {mean_accuracy * 100:.3f}%')
    return class_accuracies, mean_accuracy

def predict_with_optimized_threshold(model, data, y_test_multi_label, mlb):
    predictions = model.predict(data)
    thresholds = []  # Store optimal thresholds for each class
    for i in range(len(mlb.classes_)):
        precision, recall, th = precision_recall_curve(y_test_multi_label[:, i], predictions[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_threshold = th[np.argmax(f1_scores)]
        thresholds.append(best_threshold)
    binary_predictions = (predictions > np.array(thresholds)).astype(int)
    return binary_predictions, thresholds

def evaluate_multi_label_model(model, X_test, y_test_multi_label, mlb):
    # Make predictions on the test set with optimized thresholds
    test_predictions, optimized_thresholds = predict_with_optimized_threshold(model, X_test, y_test_multi_label, mlb)

    # Inverse transform the binary predictions to class labels
    predicted_labels = mlb.inverse_transform(test_predictions)

    # Display a sample of predictions
    sample_size = 10
    for i in range(sample_size):
        print(f"Sample {i + 1}: Predicted Labels - {predicted_labels[i]}, Actual Labels - {mlb.inverse_transform(y_test_multi_label[i:i+1])}")

    # Additional evaluation metrics
    precision = precision_score(y_test_multi_label, test_predictions, average='samples', zero_division=0)
    recall = recall_score(y_test_multi_label, test_predictions, average='samples', zero_division=0)
    f1 = f1_score(y_test_multi_label, test_predictions, average='samples')
    roc_auc = roc_auc_score(y_test_multi_label, test_predictions)
    average_precision = average_precision_score(y_test_multi_label, test_predictions)

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Average Precision: {average_precision:.3f}")

    # Plot ROC curves and Precision-Recall curves for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(mlb.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test_multi_label[:, i], test_predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(y_test_multi_label[:, i], test_predictions[:, i])
        average_precision[i] = auc(recall[i], precision[i])
    
    return test_predictions, optimized_thresholds

def main():
  # Define the list of class names
  class_names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
  # Specify custom train, validation, and test sizes
  train_size = 0.6
  val_size = 0.2
  test_size = 0.2

  # Create an instance of the CustomImageDatasetMultiLabel class
  custom_dataset_multi_label = CustomImageDatasetMultiLabel(data_dir='EuroSAT_RGB/', class_names = class_names, train_size=train_size, val_size=val_size, test_size=test_size)
        
  # Access the training, validation, and test data
  X_train, y_train = custom_dataset_multi_label.get_train_data()
  X_val, y_val = custom_dataset_multi_label.get_val_data()
  X_test, y_test = custom_dataset_multi_label.get_test_data()
  # Print the shapes of the data arrays
  print("X_train shape:", X_train.shape)
  print("y_train shape:", y_train.shape)
  print("X_val shape:", X_val.shape)
  print("y_val shape:", y_val.shape)
  print("X_test shape:", X_test.shape)
  print("y_test shape:", y_test.shape)


  # Example usage:
  X_train_augmented, y_train_augmented, X_val_augmented, y_val_augmented = augment_data(X_train, y_train, X_val, y_val, augmentation_factor=1)
  # Print the shapes of the augmented data arrays
  print("X_train_augmented shape:", X_train_augmented.shape)
  print("y_train_augmented shape:", y_train_augmented.shape)
  print("X_val_augmented shape:", X_val_augmented.shape)
  print("y_val_augmented shape:", y_val_augmented.shape)

  model, mlb, y_train_multi_label, y_val_multi_label, y_test_multi_label, history = train_multi_label_model(X_train_augmented, y_train_augmented, X_val_augmented, y_val_augmented, y_test, num_epochs=10)

  mean_average_precision_val, mean_average_precision_test, y_val_pred, y_test_pred = calculate_mean_average_precision(model, X_val, X_test, y_val_multi_label, y_test_multi_label, mlb.classes_)

  # Calculate and report accuracy on the validation set
  calculate_and_report_accuracy(y_val_multi_label, (y_val_pred > 0.5).astype(int), mlb.classes_, 'Validation Set')
  print("\n")
  # Calculate and report accuracy on the test set
  calculate_and_report_accuracy(y_test_multi_label, (y_test_pred > 0.5).astype(int), mlb.classes_, 'Test Set')

  evaluate_multi_label_model(model, X_test, y_test_multi_label, mlb)

  train_accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  # Create a figure with two subplots side by side
  plt.figure(figsize=(14, 5))

  # Plot training and validation accuracy in the first subplot
  plt.subplot(1, 2, 1)
  plt.plot(train_accuracy, label='Train Accuracy')
  plt.plot(val_accuracy, label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Accuracy over epoch')
  plt.legend()

  # Extract training and validation loss values from the history object
  train_loss = history.history['loss']
  val_loss = history.history['val_loss']

  # Plot training and validation loss in the second subplot
  plt.subplot(1, 2, 2)
  plt.plot(train_loss, label='Train Loss')
  plt.plot(val_loss, label='Validation Loss',)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss over epoch')
  plt.legend()

  # Adjust the layout for better spacing
  plt.tight_layout()

  # Show the figure with both subplots
  plt.show()

  

if __name__ == "__main__":
    main()