import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Print and log message: Loading the dataset
logging.info("Loading dataset from CSV file...")
print("Loading dataset from CSV file...")

# Load the dataset (assuming it's a CSV file)
data = pd.read_csv('/kaggle/input/connect4-data/connect4_data.csv')

# Print and log message: Preparing the data
logging.info("Preparing data for training...")
print("Preparing data for training...")

# Prepare the data
X = data.iloc[:, :42].values  # Input: the flattened 6x7 grid (42 positions)
y = data.iloc[:, 42].values   # Output: the winner (1, -1, or 0)

# Label transformation: Convert -1 -> 0, 0 -> 1, 1 -> 2
y_transformed = np.where(y == -1, 0, np.where(y == 0, 1, 2))

# Split the data into training and testing sets
logging.info("Splitting data into training and test sets...")
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

# Build the neural network model
logging.info("Building the neural network model...")
print("Building the neural network model...")

model = Sequential()
model.add(Dense(128, input_dim=42, activation='relu'))  # Hidden layer 1
model.add(Dense(64, activation='relu'))                # Hidden layer 2
model.add(Dense(3, activation='softmax'))              # Output layer (3 classes)

# Compile the model
logging.info("Compiling the model...")
print("Compiling the model...")

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
logging.info("Training the model...")
print("Training the model...")

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Log training completion
logging.info("Training complete.")
print("Training complete.")

# Save the trained model to the Kaggle working directory
output_path = '/kaggle/working/connect4_model.h5'
logging.info(f"Saving the trained model to {output_path}...")
print(f"Saving the trained model to {output_path}...")
model.save(output_path)  # Save the model to the correct directory


# Evaluate the model on the test set
logging.info("Evaluating the model on the test set...")
print("Evaluating the model on the test set...")

loss, accuracy = model.evaluate(X_test, y_test)
logging.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

# Optionally, log the training history (loss/accuracy over epochs)
logging.info(f"Training history: {history.history}")
