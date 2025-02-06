import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import keyboard
from dataclasses import dataclass
from itertools import product


@dataclass
class GestureHandler:
    model_path: str = None

    def draw_landmarks(image, results):
        """
        Draw the landmarks on the image.

        Args:
            image (numpy.ndarray): The input image.
            results: The landmarks detected by Mediapipe.

        Returns:
            numpy.ndarray: The image with drawn landmarks
        """
        # Make a copy of the image to ensure it's writable
        image = image.copy()

        # Draw landmarks for left hand if present
        if results.left_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
            )

        # Draw landmarks for right hand if present
        if results.right_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
            )

        return image


    def image_process(image, model):
        """
        Process the image and obtain sign landmarks.

        Args:
            image (numpy.ndarray): The input image.
            model: The Mediapipe holistic object.

        Returns:
            tuple: (results, processed_image) where results contains the landmarks
            and processed_image is the BGR image
        """
        # Make a copy to avoid modifying the original
        image = image.copy()

        # Convert the image from BGR to RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Process the image using the model
        # MediaPipe works better with read-only images
        image_rgb.flags.writeable = False
        results = model.process(image_rgb)
        image_rgb.flags.writeable = True

        # Convert back to BGR for OpenCV operations
        processed_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)

        return results, processed_image


    def keypoint_extraction(results):
        """
        Extract the keypoints from the sign landmarks.

        Args:
            results: The processed results containing sign landmarks.

        Returns:
            numpy.ndarray: The extracted keypoints.
        """
        # Extract the keypoints for the left hand if present, otherwise set to zeros
        lh = (
            np.array(
                [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
            ).flatten()
            if results.left_hand_landmarks
            else np.zeros(63)
        )
        # Extract the keypoints for the right hand if present, otherwise set to zeros
        rh = (
            np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            ).flatten()
            if results.right_hand_landmarks
            else np.zeros(63)
        )
        # Concatenate the keypoints for both hands
        return np.concatenate([lh, rh])


@dataclass
class VideoHandler(GestureHandler):
    video_folder: str = None
    global_timestamp: int = 0

    def create_directories(self, path:str, signs:list[str], sequences) -> None:
        # Create directories for each action, sequence, and frame in the dataset
        for action, sequence in product(signs, range(sequences)):
            try:
                os.makedirs(os.path.join(path, action, str(sequence)))
            except:
                pass


    def create_dataset(self, signs: list[str], path: str) -> None:
        # Define the number of sequences and frames to be recorded for each action
        sequences = 30
        frames = 10

        # Set the path where the dataset will be stored
        PATH = os.path.join(path)

        self.create_directories(PATH, signs, sequences)

        for video_file in os.listdir(self.video_folder):
            video_path: str = os.path.join(self.video_folder, video_file)
            if not video_file.endswith((".mp4", ".avi", ".mov")):
                continue
            print(f"Processing video: {video_file}")
            cap = cv.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_file}")
                continue
            frame_index: int = 0
            fps = cap.get(cv.CAP_PROP_FPS) or 30  # Default to 30 FPS if unknown
            # THIS LINE BELOW NEEDS TO BE INSIDE THE WHILE LOOP
            frame_index += 1
            # THESE LINES NEED ARE GOOD IN THIS INDENTATION
            # Update the global timestamp for the next video
            self.global_timestamp += int(
                cap.get(cv.CAP_PROP_FRAME_COUNT) * 1000 / fps
            )
            cap.release()
            cv.destroyAllWindows()

    def train(self) -> None:
        pass

    def run(self) -> None:

        for video_file in os.listdir(self.video_folder):
            video_path: str = os.path.join(self.video_folder, video_file)
            if not video_file.endswith((".mp4", ".avi", ".mov")):
                continue
            print(f"Processing video: {video_file}")
            cap = cv.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_file}")
                continue
            frame_index: int = 0
            fps = cap.get(cv.CAP_PROP_FPS) or 30  # Default to 30 FPS if unknown
            while True:
                success, frame = cap.read()
                if not success:
                    break
                cv.imshow("Video", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    self.stop()
                    break
                frame_index += 1
            # Update the global timestamp for the next video
            self.global_timestamp += int(cap.get(cv.CAP_PROP_FRAME_COUNT) * 1000 / fps)
            cap.release()

    def stop(self) -> None:
        cv.destroyAllWindows()


class WebCamHandler(GestureHandler):
    def run(self) -> None:
        self.create_options(self.VisionRunningMode.LIVE_STREAM, self.handle_gesture)
        cap = cv.VideoCapture(0)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Add the image dimensions to the gesture recognizer options
        self.gesture_options.base_options.frame_width = frame_width
        self.gesture_options.base_options.frame_height = frame_height

        while cap.isOpened():
            frame_index: int = 0
            fps = cap.get(cv.CAP_PROP_FPS) or 30  # Default to 30 FPS if unknown

            while True:
                success, frame = cap.read()
                if not success:
                    continue

                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # Ensure global timestamp is used
                timestamp_ms = int(frame_index * 1000 / fps)

                cv.imshow("WebCam", frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    self.stop()
                    break

                frame_index += 1

            cap.release()

    def stop(self) -> None:
        cv.destroyAllWindows()
