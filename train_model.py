from typing import Sequence 
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import os

# Disable specific optimizations (if required)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Read the data (ensure files are CSV formatted or adjust accordingly)
neutral_df = pd.read_csv("neutral.txt", delimiter=",")
grasping_df = pd.read_csv("grasping.txt", delimiter=",")
holding_df = pd.read_csv("holding.txt", delimiter=",")
gripping_df = pd.read_csv("gripping.txt", delimiter=",")
violent_df = pd.read_csv("violent.txt", delimiter=",")

# Find the maximum number of features across all datasets
all_datasets = [neutral_df, grasping_df, holding_df, gripping_df, violent_df]
max_features = max(df.shape[1] - 1 for df in all_datasets)  # Exclude the first column

# Function to process datasets and pad/truncate features
def process_dataset(dataframe, label, X, y, no_of_timesteps, max_features):
    datasets = dataframe.iloc[:, 1:].values  # Skip the first column if it's an index
    n_samples, n_features = datasets.shape

    # Pad or truncate features to match `max_features`
    if n_features < max_features:
        # Pad with zeros
        datasets = np.pad(datasets, ((0, 0), (0, max_features - n_features)), mode='constant')
    elif n_features > max_features:
        # Truncate extra features
        datasets = datasets[:, :max_features]

    for i in range(no_of_timesteps, n_samples):
        sequence = datasets[i - no_of_timesteps:i, :]
        if sequence.shape == (no_of_timesteps, max_features):  # Validate shape
            X.append(sequence)
            y.append(label)

# Prepare the feature and label lists
X = []
y = []
no_of_timesteps = 20

# Process each dataset
process_dataset(neutral_df, 0, X, y, no_of_timesteps, max_features)
process_dataset(grasping_df, 1, X, y, no_of_timesteps, max_features)
process_dataset(holding_df, 2, X, y, no_of_timesteps, max_features)
process_dataset(gripping_df, 3, X, y, no_of_timesteps, max_features)
process_dataset(violent_df, 4, X, y, no_of_timesteps, max_features)

# Convert to numpy arrays
X, y = np.array(X), np.array(y)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation="softmax"))  # 5 classes in total

# Compile the model
model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save the model
model.save("lstm-hand-grasping.h5")
