import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import Bunch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class EuroSatDataset:
    def __init__(self, data_dir, class_names, test_size=0.3, random_state=42):
        self.data_dir = os.path.abspath(data_dir)
        self.class_names = class_names
        self.test_size = test_size
        self.random_state = random_state

        self.label_binarizer = LabelBinarizer()
        self.images = Bunch()

        self.load_data()

    def load_data(self):
        X = []  # Images
        y = []  # Labels

        for class_label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith(".jpg"):
                    img = Image.open(os.path.join(class_dir, filename))
                    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]

                    X.append(img_array)
                    y.append(class_name)

        # Convert labels to one-hot encoding
        y_binarized = self.label_binarizer.fit_transform(y)

        # Split the data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y_binarized, test_size=self.test_size, random_state=self.random_state)

        # Further split the temporary set into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.random_state)

        # Assign data to class attributes
        self.images.train = np.array(X_train)
        self.images.train_labels = np.array(y_train)
        self.images.val = np.array(X_val)
        self.images.val_labels = np.array(y_val)
        self.images.test = np.array(X_test)
        self.images.test_labels = np.array(y_test)

    def get_train_data(self):
        return self.images.train, self.images.train_labels

    def get_val_data(self):
        return self.images.val, self.images.val_labels

    def get_test_data(self):
        return self.images.test, self.images.test_labels

def check_disjoint_splits(X_train, X_val, X_test):
    # Convert indices to sets for training, validation, and test sets
    train_indices_set = set(range(len(X_train)))
    val_indices_set = set(range(len(X_train), len(X_train) + len(X_val)))
    test_indices_set = set(range(len(X_train) + len(X_val), len(X_train) + len(X_val) + len(X_test)))

    # Create a set for all indices
    all_indices_set = set(range(len(X_train) + len(X_val) + len(X_test)))

    disjoint_check = (
        len(train_indices_set.intersection(val_indices_set)) == 0 and
        len(train_indices_set.intersection(test_indices_set)) == 0 and
        len(val_indices_set.intersection(test_indices_set)) == 0 and
        len(train_indices_set.union(val_indices_set, test_indices_set)) == len(all_indices_set)
    )

    return disjoint_check

def augment_data(X_train, y_train, X_val, y_val, num_augmentations=1):
    # Define data augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Lists to store augmented data and labels
    X_train_augmented = []
    X_val_augmented = []
    y_train_augmented = []
    y_val_augmented = []

    # Apply data augmentation to the training set
    for i in range(X_train.shape[0]):
        image = X_train[i]
        label = y_train[i]
        for _ in range(num_augmentations):
            augmented_image = datagen.random_transform(image)
            X_train_augmented.append(augmented_image)
            y_train_augmented.append(label)

    # Apply data augmentation to the validation set
    for i in range(X_val.shape[0]):
        image = X_val[i]
        label = y_val[i]
        for _ in range(num_augmentations):
            augmented_image = datagen.random_transform(image)
            X_val_augmented.append(augmented_image)
            y_val_augmented.append(label)

    # Convert the augmented data and labels to numpy arrays
    X_train_augmented = np.array(X_train_augmented)
    X_val_augmented = np.array(X_val_augmented)
    y_train_augmented = np.array(y_train_augmented)
    y_val_augmented = np.array(y_val_augmented)

    return X_train_augmented, y_train_augmented, X_val_augmented, y_val_augmented

def create_and_train_resnet_model(X_train, y_train, X_val, y_val, num_classes, num_epochs=10, model_save_path="task1model_single_label"):
    # Load the ResNet50 pretrained model without top layers
    base_model = ResNet50(weights='imagenet', include_top=False)

    # Modify the model architecture for single-label prediction
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    # Freeze the pretrained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model with categorical cross-entropy loss for single-label classification
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))

    # Save the model
    model.save(model_save_path)

    return model, history

def calculate_mean_avg_precision(X_train_augmented, y_train_augmented, X_val_augmented, y_val_augmented, X_test, y_test, class_names):
    # Flatten y_train to make it a 1D array
    y_train_flat = np.argmax(y_train_augmented, axis=1)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train_augmented.reshape(len(X_train_augmented), -1), y_train_flat)

    # Validate the model
    y_val_pred_prob = model.predict_proba(X_val_augmented.reshape(len(X_val_augmented), -1))

    # Test the model
    y_test_pred_prob = model.predict_proba(X_test.reshape(len(X_test), -1))

    # Calculate and report average precision for each class on the validation set
    print("Average Precision on Validation Set:")
    mean_avg_precision_val = 0.0
    class_avg_precisions_val = []
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_val_augmented[:, i], y_val_pred_prob[:, i])
        avg_precision = auc(recall, precision)
        class_avg_precisions_val.append(avg_precision)
        mean_avg_precision_val += avg_precision
        print(f"{class_name}: {avg_precision * 100:.3f}%")

    mean_avg_precision_val /= len(class_names)
    print(f"\nMean Average Precision on Validation Set: {mean_avg_precision_val * 100:.3f}%")

    # Calculate and report average precision for each class on the test set
    print("\nAverage Precision on Test Set:")
    mean_avg_precision_test = 0.0
    class_avg_precisions_test = []
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_test_pred_prob[:, i])
        avg_precision = auc(recall, precision)
        class_avg_precisions_test.append(avg_precision)
        mean_avg_precision_test += avg_precision
        print(f"{class_name}: {avg_precision * 100:.3f}%")

    mean_avg_precision_test /= len(class_names)
    print(f"Mean Average Precision on Test Set: {mean_avg_precision_test * 100:.3f}%")

    return y_val_pred_prob, y_test_pred_prob

def calculate_accuracies(y_pred_prob, y_true, class_names):
    # Convert predicted probabilities to class labels
    y_pred_class = np.argmax(y_pred_prob, axis=1)

    # Calculate and report accuracy for each class
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        class_accuracy = accuracy_score(y_true[:, i], y_pred_class)
        class_accuracies[class_name] = class_accuracy * 100

    # Calculate mean accuracy
    mean_accuracy = np.mean(list(class_accuracies.values()))

    return class_accuracies, mean_accuracy

def main():
    # Example usage:
    class_names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
    eurosat_dataset = EuroSatDataset(data_dir='EuroSAT_RGB', class_names=class_names)

    X_train, y_train = eurosat_dataset.get_train_data()
    X_val, y_val = eurosat_dataset.get_val_data()
    X_test, y_test = eurosat_dataset.get_test_data()

    # Print shapes of the datasets
    print("Train data shape:", X_train.shape)
    print("Train labels shape:", y_train.shape)

    print("Validation data shape:", X_val.shape)
    print("Validation labels shape:", y_val.shape)

    print("Test data shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)


    #check disjoint
    result = check_disjoint_splits(X_train, X_val, X_test)
    if result:
        print("Data splits are disjoint and contain no duplicates.")
    else:
        print("Error: Data splits are not disjoint or contain duplicates.")

    # Usage for data aug
    X_train_augmented, y_train_augmented, X_val_augmented, y_val_augmented = augment_data(X_train, y_train, X_val, y_val, num_augmentations=1)
    # Print the shapes of the augmented data arrays
    print("X_train_augmented shape:", X_train_augmented.shape)
    print("y_train_augmented shape:", y_train_augmented.shape)
    print("X_val_augmented shape:", X_val_augmented.shape)
    print("y_val_augmented shape:", y_val_augmented.shape)
    print('='*50)
    num_classes = 10  
    model, history = create_and_train_resnet_model(X_train_augmented, y_train_augmented, X_val_augmented, y_val_augmented, num_classes, num_epochs=10)
    print('='*50)
    y_val_pred_prob, y_test_pred_prob = calculate_mean_avg_precision(X_train_augmented, y_train_augmented, X_val_augmented, y_val_augmented, X_test, y_test, class_names)
    print('='*50)

    # Usage example:
    # Call the calculate_mean_avg_precision function
    class_accuracies_val, mean_accuracy_val = calculate_accuracies(y_val_pred_prob, y_val, class_names)
    class_accuracies_test, mean_accuracy_test = calculate_accuracies(y_test_pred_prob, y_test, class_names)
    print("Accuracy per Class on Validation Set:")
    for class_name, accuracy in class_accuracies_val.items():
        print(f"{class_name}: {accuracy:.3f}%")
    print(f"Mean Accuracy on Validation Set: {mean_accuracy_val:.3f}%")

    print("\nAccuracy per Class on Test Set:")
    for class_name, accuracy in class_accuracies_test.items():
        print(f"{class_name}: {accuracy:.3f}%")
    print(f"Mean Accuracy on Test Set: {mean_accuracy_test:.3f}%")
    print('='*50)

    # Extract training and validation accuracy values from the history object
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
