import tensorflow as tf
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder

current_directory = Path(__file__).resolve().parent

# Load the train.csv file
train_df = pd.read_csv(f'{current_directory}/model_train/train.csv')

# Add leading zeros to the IDs in train_df
train_df['Id'] = train_df['Id'].apply(lambda x: str(x).zfill(4))

# Define the image directory
image_directory = f'{current_directory}/model_train/images/'

# Preprocess the training data
train_images = []
train_labels = []
label_encoder = LabelEncoder()

for _, row in train_df.iterrows():
    # Load the image using PIL
    image_path = image_directory + row['Id'] + '.jpg'
    image = Image.open(image_path)

    # Resize the image to the desired size
    image = image.resize((48, 48))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the image array
    image_array = image_array / 255.0

    # Add the image array to the list
    train_images.append(image_array)

    # Add the corresponding label to the list
    train_labels.append(row['label'])

# Convert the lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Encode the string labels to integer labels
train_labels = label_encoder.fit_transform(train_labels)

# Convert labels to one-hot encoded format
num_classes = len(label_encoder.classes_)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)

# Create a TensorFlow dataset from the numpy arrays
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

# Shuffle and batch the dataset
train_dataset = train_dataset.shuffle(len(train_df)).batch(32)

# Load the test.csv file
test_df = pd.read_csv(f'{current_directory}/model_train/test.csv')

# Add leading zeros to the IDs in test_df
test_df['Id'] = test_df['Id'].apply(lambda x: str(x).zfill(4))

# Preprocess the test data
test_images = []
for _, row in test_df.iterrows():
    # Load the image using PIL
    image_path = image_directory + row['Id'] + '.jpg'
    image = Image.open(image_path)

    # Resize the image to the desired size
    image = image.resize((48, 48))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the image array
    image_array = image_array / 255.0

    # Add the image array to the list
    test_images.append(image_array)

# Convert the list of images to a numpy array
test_images = np.array(test_images)

# Create a TensorFlow dataset from the numpy array
test_dataset = tf.data.Dataset.from_tensor_slices(test_images)

# Batch the dataset
test_dataset = test_dataset.batch(32)

# Load the pre-trained CNN model
model = tf.keras.Sequential()

# Add convolutional layers
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the output from the convolutional layers
model.add(tf.keras.layers.Flatten())

# Add a dense layer for classification
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))  # Use num_classes instead of hardcoding 6

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    epochs=25,
    validation_data=test_dataset
)

# Save the trained model
model.save(f'{current_directory}/model_train/model/trained_model.h5')
