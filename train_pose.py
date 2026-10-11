from typing import Sequence, Tuple
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Disable specific optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging


def load_and_process_data(filepath: str, label: int, no_of_timesteps: int = 20) -> Tuple[Sequence[np.ndarray], Sequence[int]]:
    """
    Load and process data from a single file
    
    Args:
        filepath: Path to the CSV file
        label: Label to assign to the data (0 for neutral, 1 for violent)
        no_of_timesteps: Number of timesteps for LSTM input
        
    Returns:
        Tuple of features (X) and labels (y) lists
    """
    try:
        logging.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Ensure data has at least one row and expected dimensions
        if df.shape[1] < 2:
            raise ValueError(f"File {filepath} does not contain enough columns for processing.")
        
        datasets = df.iloc[:, 1:].values  # Skip the first column (assumed index)
        n_samples = len(datasets)
        
        X = []
        y = []
        
        for i in range(no_of_timesteps, n_samples):
            X.append(datasets[i - no_of_timesteps:i, :])
            y.append(label)
        
        logging.info(f"Loaded {len(X)} sequences from {filepath}.")
        return X, y
    except Exception as e:
        logging.error(f"Error loading file {filepath}: {str(e)}")
        raise


def train_model():
    """Train and save the LSTM model"""
    try:
        # Parameters
        no_of_timesteps = 20

        # Initialize lists for features and labels
        X = []
        y = []

        # Load and process neutral data (label 0)
        logging.info("Loading neutral data...")
        neutral_X, neutral_y = load_and_process_data("neutral.txt", 0, no_of_timesteps)
        X.extend(neutral_X)
        y.extend(neutral_y)

        # Load and process violent data (label 1)
        logging.info("Loading violent data...")
        violent_X, violent_y = load_and_process_data("violent.txt", 1, no_of_timesteps)
        X.extend(violent_X)
        y.extend(violent_y)

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        if X.size == 0 or y.size == 0:
            raise ValueError("No valid data loaded. Check the content and format of your input files.")

        logging.info(f"Data shapes - X: {X.shape}, y: {y.shape}")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(1, activation="sigmoid")  # Binary classification
        ])

        # Compile model
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Model summary
        model.summary()

        # Train the model
        logging.info("Training model...")
        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test)
        logging.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        # Save the model
        model.save("lstm-model.h5")
        logging.info("Model saved as lstm-model.h5")

        return history, model

    except Exception as e:
        logging.error(f"Error in training process: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        history, model = train_model()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
