import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1)  # Detect one hand
        self.mpDraw = mp.solutions.drawing_utils
        self.label = "not grasped"

    def calculate_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def check_grasped(self, landmarks):
        """Determine if the hand is in a grasped state."""
        # Thumb and index finger tips
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        # Distance between thumb and index tip
        thumb_index_distance = self.calculate_distance(thumb_tip[:2], index_tip[:2])
        index_middle_distance = self.calculate_distance(index_tip[:2], middle_tip[:2])
        
        # Define thresholds for "grasped"
        if thumb_index_distance < 0.05 and index_middle_distance < 0.05:
            self.label = "grasped"
        else:
            self.label = "not grasped"

    def draw_landmark_on_image(self, frame, results):
        """Draw landmarks on the frame."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return frame

    def draw_bounding_box_and_label(self, frame, landmarks):
        """Draw bounding box and label."""
        if landmarks:
            x_min = int(min(lm[0] for lm in landmarks) * frame.shape[1])
            y_min = int(min(lm[1] for lm in landmarks) * frame.shape[0])
            x_max = int(max(lm[0] for lm in landmarks) * frame.shape[1])
            y_max = int(max(lm[1] for lm in landmarks) * frame.shape[0])
            
            # Draw bounding box and label
            color = (0, 255, 0) if self.label == "not grasped" else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"Status: {self.label}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    def run(self):
        """Main loop for detecting hand state."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open camera")

        cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Detection", 1200, 1000)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract 3D landmarks
                        landmarks = [
                            [lm.x, lm.y, lm.z]
                            for lm in hand_landmarks.landmark
                        ]
                        # Check if hand is grasped
                        self.check_grasped(landmarks)
                        # Draw annotations
                        frame = self.draw_landmark_on_image(frame, results)
                        self.draw_bounding_box_and_label(frame, landmarks)

                # Display frame
                cv2.imshow("Hand Detection", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = HandDetector()
    detector.run()
