import os
import cv2 as cv
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn import metrics
from sklearn.model_selection import train_test_split
from core.processor import MediaPipeProcessor


class GestureHandler(MediaPipeProcessor):
    def __init__(
        self,
        confidence: float = 0.75,
        mode: str = "holistic",
        data_parent_folder: str = None,
    ):
        """The Base Class for Sign Language data creation.

        Args:
            confidence (float): The minimun tracking and detection confidence of the MediaPipeProcessor model. Defaults to 0.75.
            mode (str): The desired Mediapipe model. Defaults to 'holistic'.
            data_parent_folder (str): The folder that contains all the data
            that will be used in the dataset creation.

        **NOTE** This class is ONLY a collection of methods that both ImageHandler and
        VideoHandler use.
        """
        self.data_parent_folder = data_parent_folder
        super().__init__(confidence, mode)

    def get_file_index(self, file_name: str) -> str:
        """Returns the numeric value in the given file name.
        This numeric value refers to the index of the video of
        some label.
        (e. g. ADIOS1 -> 1, ADIOS2 -> 2, etc.)

        Args:
            file_name (str)

        Returns:
            str: The file index.
        """
        index = ""
        for character in file_name:
            if character == ".":
                break
            elif character.isnumeric():
                index += character
        return index

    def get_label_name(self, file_name: str) -> str:
        """Returns the label name of the given file name.
        The label is associated with the gesture shown in the file.

        Args:
            file_name (str)

        Returns:
            str:
        """
        label = ""
        for character in file_name:
            if character == ".":
                break
            elif character.isalpha() or character == "_":
                label += character
        return label

    def create_dataset_directories(self, path: str) -> None:
        """Creates all the needed directories for the dataset inside the given path.

        Args:
            path (str): Where the directories should be created.
        """
        with os.scandir(self.data_parent_folder) as entries:
            for entry in entries:
                if entry.is_dir():
                    os.makedirs(os.path.join(path, entry.name), exist_ok=True)


    def dataset_directories_already_created(self, path: str) -> bool:
        """Return True if the dataset's needed directories are already created
        inside the given path, False otherwise.

        Args:
            path (str): Where the dataset's needed directories should be.

        Returns:
            bool: True if already created, False otherwise.
        """
        return len(os.listdir(path)) == len(os.listdir(self.data_parent_folder))

    def stop(self) -> None:
        cv.destroyAllWindows()


class ImageHandler(GestureHandler):
    def __init__(
        self,
        confidence: float = 0.75,
        mode: str = "hands",
        data_parent_folder: str = None,
    ):
        """Sign Language data pipeline for images.

        Args:
            confidence (float): The desired confidence of the MediaPipeProcessor model. Defaults to 0.75.
            mode (str): The desired Mediapipe model. Defaults to 'hands'.
            data_parent_folder (str): The folder that contains all the data
            that will be used in the dataset creation.
        
        NOTE: mode is set to 'hands' because usually only hands are used to sign letters (which do not need spatial
        nor temporal context)
        """
        super().__init__(confidence, mode, data_parent_folder)

    def create_dataset(self, path: str) -> None:
        """
        Creates the dataset for the model training in the given path.

        Args:
            path (str): Where the dataset will be created.
        """
        if not self.dataset_directories_already_created(path):
            self.create_dataset_directories(path)

        image_folders = os.scandir(self.data_parent_folder)

        for image_folder in image_folders:
            if image_folder.is_dir():
                image_folder_path_dirs = os.scandir(
                    image_folder.path
                )
                for image_file in image_folder_path_dirs:
                    if image_file.is_file():
                        image_path = image_file.path
                        print(f"Processing image: {image_file.name}")
                        frame = cv.imread(image_path)
                        if frame is None:
                            print(f"Couldn't open {image_file.name}")
                            continue
                        resized_frame = cv.resize(frame, (640, 480))
                        # Process image and get results
                        results, processed_image = self.image_process(resized_frame)
                        if not self.are_results_valid(results):
                            print(f"No valid landmarks given the criteria of {self.mode} model")
                            continue
                        # Draw landmarks and display
                        display_image = self.draw_landmarks(processed_image, results)
                        # Extract the landmarks from both hands and save them in arrays
                        keypoints: np.ndarray = self.keypoint_extraction(results)
                        frame_path = os.path.join(
                            path,
                            self.get_label_name(image_file.name),
                            f"{self.get_file_index(image_file.name)}.npy",
                        )
                        save_dir = os.path.dirname(frame_path)
                        os.makedirs(save_dir, exist_ok=True)
                        print(f"Saving keypoints shape {keypoints.shape} to {frame_path}")
                        if os.path.exists(frame_path):
                            loaded = np.load(frame_path)
                            print(f"Verified save: loaded shape {loaded.shape}")
                        np.save(frame_path, keypoints)
                        resized_frame = cv.resize(display_image, (960, 540))
                        cv.imshow("Image", resized_frame)
                        if cv.waitKey(1) & 0xFF == ord("q"):
                            self.stop()
                            break
        self.stop()
                

    def train(
        self, dataset_path: str, model_path: str = "models/letters_model.keras"
    ) -> None:
        """Creates a trained model using the dataset inside the dataset path.

        Args:
            dataset_path (str): Where the dataset is.
            model_path (str): Where the resulting model should be stored.
        """
        labels = os.listdir(self.data_parent_folder)
        signs = np.array(labels)

        landmarks, labels_integers = self.load_frame(dataset_path, labels)

        # X, Y = landmarks, labels_integers

        # Split the data into training and testing sets
        (
            training_landmarks,
            testing_landmarks,
            training_labels_integers,
            testing_labels_integers,
        ) = train_test_split(
            landmarks, labels_integers, test_size=0.10, random_state=34
        )

        # Define the model architecture
        model = Sequential()
        model.add(
            Dense(
                128,
                activation="relu",
                input_shape=(126,),
            )
        )
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(signs.shape[0], activation="softmax"))

        # Compile the model with Adam optimizer and categorical cross-entropy loss
        model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )
        # Train the model
        model.fit(training_landmarks, training_labels_integers, epochs=100)

        # Save the trained model
        model.save(model_path)

        # Make predictions on the test set
        predictions = np.argmax(model.predict(testing_landmarks), axis=1)
        # Get the true labels from the test set
        test_labels = np.argmax(testing_labels_integers, axis=1)

        # Calculate the accuracy of the predictions
        accuracy = metrics.accuracy_score(test_labels, predictions)
        print(accuracy)

    def load_frame(self, dataset_path: str, labels: list[str]):
        """Loads the frame's data into an numpy array for model training.

        Args:
            dataset_path (str): Where the dataset is.
            labels (list[str]): The words that the model will learn.

        Returns:
            tuple: The frames data and the labels integers.
        """
        landmarks, labels_integers = [], []
        label_map = {label: num for num, label in enumerate(labels)}
        
        for label in os.scandir(dataset_path):
            if label.is_dir():
                for file in os.scandir(label.path):
                    if file.is_file() and file.endswith(".npy"):
                        frame_data = np.load(os.path.join(dataset_path, label, file))
                        landmarks.append(frame_data)
                        labels_integers.append(label_map[label])

        return np.array(landmarks), to_categorical(labels_integers).astype(int)


class VideoHandler(GestureHandler):
    def __init__(
        self,
        confidence: float = 0.75,
        mode: str = "holistic",
        data_parent_folder: str = None,
    ):
        """Sign Language data pipeline for videos.

        Args:
            confidence (float): The desired confidence of the MediaPipeProcessor model. Defaults to 0.75.
            mode (str): The desired Mediapipe model. Defaults to 'holistic'.
            data_parent_folder (str): The folder that contains all the data
            that will be used in the dataset creation.

        NOTE: mode is set to 'holistic' because sign language videos normally sign words or concepts,
        which use things like pose, face and require spatial and temporal context.
        """
        super().__init__(confidence, mode, data_parent_folder)

    def create_dataset(self, path: str) -> None:
        """
        Creates the dataset for the model training in the given path.

        Args:
            path (str): Where the dataset will be created.
        """
        if not self.dataset_directories_already_created(path):
            self.create_dataset_directories(path)

        video_folders = os.scandir(self.data_parent_folder)

        for video_folder in video_folders:
            if video_folder.is_dir():
                video_folder_path_dirs = os.scandir(
                    video_folder.path
                )
                for video_file in video_folder_path_dirs:
                    if video_file.is_file():
                        video_path = video_file.path
                        if not video_file.name.endswith((".mp4", ".avi", ".mov")):
                            continue
                        print(f"Processing video: {video_file.name}")
                        cap = cv.VideoCapture(video_path)
                        if not cap.isOpened():
                            print(f"Failed to open video: {video_file.name}")
                            continue
                        frame_index: int = 0
                        while True:
                            success, frame = cap.read()
                            if not success or frame is None:
                                break
                            print("PROCESSING FRAME:", frame_index)
                            resized_frame = cv.resize(frame, (640, 480))
                            # Process image and get results
                            results, processed_image = self.image_process(resized_frame)
                            if not self.are_results_valid(results):
                                print(
                                    f"No valid landmarks given the criteria of {self.mode} model"
                                )
                                frame_index += 1
                                continue
                            # Draw landmarks and display
                            display_image = self.draw_landmarks(processed_image, results)
                            # Extract the landmarks from both hands and save them in arrays
                            keypoints: np.ndarray = self.keypoint_extraction(results)
                            frame_path = os.path.join(
                                path,
                                self.get_label_name(video_file.name),
                                f"{self.get_file_index(video_file.name)}_frame_{frame_index}.npy",
                            )
                            save_dir = os.path.dirname(frame_path)
                            os.makedirs(save_dir, exist_ok=True)
                            print(f"Saving keypoints shape {keypoints.shape} to {frame_path}")
                            if os.path.exists(frame_path):
                                loaded = np.load(frame_path)
                                print(f"Verified save: loaded shape {loaded.shape}")
                            np.save(frame_path, keypoints)
                            frame_index += 1
                            resized_frame = cv.resize(display_image, (960, 540))
                            cv.imshow("Video", resized_frame)
                            if cv.waitKey(1) & 0xFF == ord("q"):
                                self.stop()
                                break
                        cap.release()

    def create_sequences(
        self, dataset_path: str, labels: list[str], sequence_length: int = 10
    ):
        """Creates overlapping sequences from frame data and the labels integers for training.

        Args:
            dataset_path (str): Where the dataset is.
            labels (list[str]): The words that the model will learn.
            sequence_length (int): The length of the overlapping sequences. Defaults to 10.

        Returns:
            tuple: The overlapping sequences and the labels integers.
        """
        landmarks, labels_integers = [], []
        # Create a label map to map each sign label to a numeric value
        label_map = {label: num for num, label in enumerate(labels)}
        
        for label in os.scandir(dataset_path):
            all_frames = []
            if label.is_dir():
                for file in os.scandir(label.path):
                    if file.is_file() and file.endswith(".npy"):
                        frame_data = np.load(file.path)
                        all_frames.append(frame_data)

            # Create sliding window sequences
            for i in range(len(all_frames) - sequence_length + 1):
                sequence = all_frames[i : i + sequence_length]
                landmarks.append(sequence)
                labels_integers.append(label_map[label])

        return np.array(landmarks), to_categorical(labels_integers).astype(int)

    def train(
        self, dataset_path: str, model_path: str = "models/words_model.keras"
    ) -> None:
        """Creates a trained model using the dataset inside the dataset path.

        Args:
            dataset_path (str): Where the dataset is.
            model_path (str): Where the resulting model should be stored.
        """
        labels = os.listdir(self.data_parent_folder)
        signs = np.array(labels)
        SEQUENCE_LENGTH = 10  # Must match model's expected input

        landmarks, labels_integers = self.create_sequences(
            dataset_path, labels, SEQUENCE_LENGTH
        )

        # X, Y = landmarks, labels_integers

        # Split the data into training and testing sets
        (
            training_landmarks,
            testing_landmarks,
            training_labels_integers,
            testing_labels_integers,
        ) = train_test_split(
            landmarks, labels_integers, test_size=0.10, random_state=34
        )

        # Define the model architecture
        model = Sequential()
        model.add(
            LSTM(
                32,
                return_sequences=True,
                activation="relu",
                input_shape=(SEQUENCE_LENGTH, 126),
            )
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
        model.fit(training_landmarks, training_labels_integers, epochs=100)

        # Save the trained model
        model.save(model_path)

        # Make predictions on the test set
        predictions = np.argmax(model.predict(testing_landmarks), axis=1)
        # Get the true labels from the test set
        test_labels = np.argmax(testing_labels_integers, axis=1)

        # Calculate the accuracy of the predictions
        accuracy = metrics.accuracy_score(test_labels, predictions)
        print(accuracy)
