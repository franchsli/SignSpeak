import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import metrics
from keras.models import Sequential
from keras.layers import LSTM, Dense


@dataclass
class GestureHandler:
    model_path: str = None

    def get_file_index(self, file_name: str) -> str:
        index = ""
        for character in file_name:
            if character == ".":
                break
            elif character.isnumeric():
                index += character
        return index

    def get_label_name(self, file_name: str) -> str:
        label = ""
        for character in file_name:
            if character == ".":
                break
            elif character.isalpha() or character == "_":
                label += character
        return label

    def needed_landmarks_present(self, results) -> bool:
        pose = results.pose_landmarks
        left_hand = results.left_hand_landmarks
        right_hand = results.right_hand_landmarks
        return (pose and left_hand) or (pose and right_hand)

    def draw_landmarks(self, image, results):
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

        # Draw pose landmarks
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS
            )

        # Draw landmarks for left hand if present
        if results.left_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
            )

        # Draw landmarks for right hand if present
        if results.right_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
            )

        return image

    def image_process(self, image, model):
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

    def keypoint_extraction(self, results):
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

    def create_directories(self, path: str) -> None:
        parent_folder = os.path.dirname(self.video_folder)
        for label in os.listdir(parent_folder):
            os.makedirs(os.path.join(path, label), exist_ok=True)

    def directories_already_created(self, path) -> bool:
        parent_folder = os.path.dirname(self.video_folder)
        return len(os.listdir(path)) == len(os.listdir(parent_folder))

    def create_dataset(self, path: str) -> None:
        if not self.directories_already_created(path):
            self.create_directories(path)

        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.75, min_tracking_confidence=0.75
        ) as holistic:
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

                    print("PROCESSING FRAME:", frame_index)

                    # rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    # Ensure global timestamp is used
                    # timestamp_ms = int(self.global_timestamp + (frame_index * 1000 / fps))
                    # Process image and get results
                    results, processed_image = self.image_process(frame, holistic)

                    if not self.needed_landmarks_present(results):
                        print(f"Not enough landmarks in {frame_index}, skipping...")
                        frame_index += 1
                        continue

                    # Draw landmarks and display
                    display_image = self.draw_landmarks(processed_image, results)

                    # Extract the landmarks from both hands and save them in arrays
                    keypoints: np.ndarray = self.keypoint_extraction(results)

                    frame_path = os.path.join(
                        path,
                        self.get_label_name(video_file),
                        f"{self.get_file_index(video_file)}_frame_{frame_index}.npy",
                    )
                    save_dir = os.path.dirname(frame_path)
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"Saving keypoints shape {keypoints.shape} to {frame_path}")
                    if os.path.exists(frame_path):
                        loaded = np.load(frame_path)
                        print(f"Verified save: loaded shape {loaded.shape}")
                    np.save(frame_path, keypoints)

                    frame_index += 1

                    cv.imshow("Video", display_image)

                    if cv.waitKey(1) & 0xFF == ord("q"):
                        self.stop()
                        break

                # Update the global timestamp for the next video
                self.global_timestamp += int(
                    cap.get(cv.CAP_PROP_FRAME_COUNT) * 1000 / fps
                )
                cap.release()

    def train(self, path: str) -> None:
        parent_folder = os.path.dirname(self.video_folder)
        labels = os.listdir(parent_folder)
        signs = np.array(labels)
        # Create a label map to map each action label to a numeric value
        label_map = {label: num for num, label in enumerate(labels)}

        # Initialize empty lists to store landmarks and labels
        landmarks, labels_integers = [], []

        # Iterate over actions and sequences to load landmarks and corresponding labels
        for label in os.listdir(path):
            temp = []
            for binary_file in os.listdir(os.path.join(path, label)):
                if binary_file.endswith(".npy"):
                    x = os.listdir(path)
                    npy = np.load(os.path.join(path, label, binary_file))
                    temp.append(npy)
            landmarks.append(temp)
            labels_integers.append(label_map[label])

        # Convert landmarks and labels to numpy arrays
        X, Y = np.array(landmarks), to_categorical(labels_integers).astype(int)

        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.10, random_state=34, stratify=Y
        )

        # Define the model architecture
        model = Sequential()
        model.add(
            LSTM(32, return_sequences=True, activation="relu", input_shape=(10, 126))
        )
        model.add(LSTM(64, return_sequences=True, activation="relu"))
        model.add(LSTM(32, return_sequences=False, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(signs.shape[0], activation="softmax"))

        # Compile the model with Adam optimizer and categorical cross-entropy loss
        model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )
        # Train the model
        model.fit(X_train, Y_train, epochs=100)

        # Save the trained model
        model.save("models")

        # Make predictions on the test set
        predictions = np.argmax(model.predict(X_test), axis=1)
        # Get the true labels from the test set
        test_labels = np.argmax(Y_test, axis=1)

        # Calculate the accuracy of the predictions
        accuracy = metrics.accuracy_score(test_labels, predictions)
        print(accuracy)

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
