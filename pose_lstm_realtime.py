import cv2
import mediapipe as mp
import numpy as np
import logging
import time
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)

class PoseDetector:
    def __init__(self):
        self.setup_mediapipe()
        self.setup_camera()
        self.previous_landmarks = None
        self.label = "No Violence Detected"
        self.velocity_threshold = 0.6  # Threshold for high-speed hand movement
        self.last_violence_time = 0  # Timestamp of the last violence detection
        self.label_duration = 1.2  # Time (in seconds) to keep the label visible

    def setup_mediapipe(self):
        """Initialize MediaPipe for pose detection."""
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True
        )
        self.mpDraw = mp.solutions.drawing_utils

    def setup_camera(self):
        """Initialize the webcam."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open the camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def calculate_hand_velocity(self, current_landmarks, previous_landmarks):
        """Calculate the velocity of hand landmarks across frames."""
        if not previous_landmarks:
            return 0  # No movement if no previous frame

        # Indices of hand landmarks in Mediapipe (right and left wrist)
        hand_indices = [15, 16]  # Right wrist and left wrist
        velocities = []

        for idx in hand_indices:
            x1, y1, z1 = current_landmarks[idx][:3]
            x2, y2, z2 = previous_landmarks[idx][:3]
            velocity = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            velocities.append(velocity)

        return max(velocities)  # Return the maximum velocity

    def process_frame(self, frame):
        """Process a single frame for pose detection and hand movement analysis."""
        try:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = [
                    [lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark
                ]

                # Calculate hand velocity
                hand_velocity = self.calculate_hand_velocity(landmarks, self.previous_landmarks)
                self.previous_landmarks = landmarks

                # Detect violence based on high-speed movement
                if hand_velocity > self.velocity_threshold:
                    self.label = "VIOLENCE DETECTED!"
                    self.last_violence_time = time.time()  # Update the timestamp
                elif time.time() - self.last_violence_time < self.label_duration:
                    self.label = "VIOLENCE DETECTED!"  # Keep the label for 2 seconds
                else:
                    self.label = "No Violence Detected"

                # Choose label color
                label_color = (0, 0, 255) if "VIOLENCE" in self.label else (0, 255, 0)

                # Draw pose landmarks
                self.mpDraw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mpDraw.DrawingSpec(color=label_color, thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mpDraw.DrawingSpec(color=label_color, thickness=2)
                )

                # Display labels
                cv2.rectangle(frame, (10, 10), (300, 80), label_color, -1)
                cv2.putText(frame, self.label,
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

            return frame
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            return frame

    def run(self):
        """Run the real-time hand movement and violence detection loop."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logging.warning("Failed to capture frame.")
                    continue

                processed_frame = self.process_frame(frame)
                cv2.imshow("Hand Movement Detection", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logging.error(f"Runtime error: {str(e)}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    try:
        detector = PoseDetector()
        detector.run()
    except RuntimeError as e:
        logging.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()