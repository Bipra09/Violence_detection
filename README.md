# Violence Detection

Real-time violence detection using MediaPipe pose/hand landmarks and an LSTM model.

## How it works
- MediaPipe extracts body pose / hand landmarks from webcam frames
- Landmark sequences are fed into an LSTM classifier
- Violent actions trigger an alert sound (`alert.wav`)

## Project structure
- `pose_data_generation.py`: collect pose landmark data
- `hands_data_generation.py`: collect hand landmark data
- `train_pose.py`, `train_model.py`: train LSTM models
- `pose_lstm_realtime.py`: real-time pose-based detection
- `hands_lstm_realtime.py`: real-time hand-based detection
- `*.txt`: collected landmark datasets (violent, neutral, resting, etc.)
- `*.h5`, `*.keras`: trained models

## Setup
```
python -m venv env
env\Scripts\activate
pip install tensorflow mediapipe opencv-python numpy
```

## Usage
```
python pose_data_generation.py
python train_pose.py
python pose_lstm_realtime.py
```

## Tech stack
Python, TensorFlow/Keras, MediaPipe, OpenCV, NumPy

## Author
[Bipra09](https://github.com/Bipra09)
