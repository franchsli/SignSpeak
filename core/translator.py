import cv2 as cv
from numpy import array, amax, argmax, newaxis
from numpy.typing import NDArray
from keras.models import load_model, Model
from PIL import ImageDraw, ImageFont, Image
from .processor import MediaPipeProcessor
from .text_processor import TextProcessor


class SignLanguageTranslator:
    def __init__(
        self,
        mediapipe_confidence: float = 0.75,
        language: str = None,
    ):
        """Sign language translation class.

        Args:
            mediapipe_confidence (float, optional): The minimun detection and tracking confidence
            that the MediaPipe model will have. Defaults to 0.75.
            language (str, optional): The language's code that will be checked to correct
            the text. If no language is given, the translator won't be corrected. Defaults to None.
        """
        self.mediapipe_confidence = mediapipe_confidence
        self.text_processor = TextProcessor(language) if language else None
        self.loaded_models = {}

    def translate_video(
        self,
        video_input: str | int = 0,
        display_in_real_time: bool = True,
        model_name: str = "words",
    ) -> str:
        """Translates the sign language found in the given video.

        Args:
            video_input (str | int, optional): The path of the video file to be translated or 0 for the webcam. Defaults to 0.
            display_in_real_time (bool, optional): Whether if the translation should be displayed as it's being translated or not.
            Defaults to True. Note that this variable will be True if the video_input is the webcam.
            model_name (str, optional): The name of the model loaded. Defaults to 'words'.

        Returns:
            str: The corrected translation if it was corrected successfully, the raw translation if not.

        NOTE: The real-time display shows uncorrected translation for immediate feedback.
                For grammatically corrected text, use the returned translation.
        """
        # set the parameter to true if the webcam is being translated
        display_in_real_time = True if video_input == 0 else display_in_real_time
        # Initialize the variables neeeded
        self.processor = MediaPipeProcessor(self.mediapipe_confidence)
        prediction_model, prediction_model_signs = self.get_model(model_name)
        keypoints, last_prediction = [], ""
        prediction_history = []
        sentence = ""
        self.display_history = []
        self.display_sentence = ""
        self.lines_counter = 1
        # Initialize constant variables
        CONFIDENCE_THRESHOLD = 0.7
        DESIRED_FRAME_WIDTH = 960
        DESIRED_FRAME_HEIGHT = 540
        EXPECTED_MODEL_KEYPOINTS_COUNT = 10
        # Access the video input and check if it's opened successfully
        cap = cv.VideoCapture(video_input)
        if not cap.isOpened():
            if video_input:
                print(f"Cannot access the given video input ({video_input}).")
            else:
                print("Cannot access the camera.")
            return

        window_created = False

        while cap.isOpened():
            success, image = cap.read()
            if not success or image is None:
                break
            resized_frame = cv.resize(
                image, (DESIRED_FRAME_WIDTH, DESIRED_FRAME_HEIGHT)
            )
            # Process the image and obtain sign landmarks
            results, processed_image = self.processor.process_image(resized_frame)
            if not self.processor.are_results_valid(results):
                print(
                    f"No valid landmarks given the criteria of {self.processor.mode} model"
                )
                key = cv.waitKey(1) & 0xFF
                # Check if "q" key was pressed or the "Translation" window was closed and break the loop
                if key == ord("q") or (
                    window_created and self._is_translation_window_closed()
                ):
                    break
                continue
            # Draw the sign landmarks on the image
            frame_with_landmarks = self.processor.draw_landmarks(
                processed_image, results
            )
            # Extract keypoints from the pose landmarks
            keypoints.append(self.processor.keypoint_extraction(results))
            # Check if 10 frames have been accumulated
            if len(keypoints) == EXPECTED_MODEL_KEYPOINTS_COUNT:
                # Convert keypoints list to a numpy array
                keypoints = array(keypoints)
                # Make a prediction on the keypoints using the loaded model
                prediction = prediction_model.predict(keypoints[newaxis, :, :])
                # Clear the keypoints list for the next set of frames
                keypoints = []
                # Check if the maximum prediction value is above 0.7
                if amax(prediction) > CONFIDENCE_THRESHOLD:
                    predicted_class = prediction_model_signs[argmax(prediction)]
                    if predicted_class != last_prediction:
                        prediction_history.append(predicted_class)
                        self.display_history.append(predicted_class)
                        last_prediction = predicted_class
                        # update the sentence with a space if it's not the first one
                        if sentence:
                            sentence += f" {predicted_class}"
                            self.display_sentence += f" {predicted_class}"
                        else:
                            sentence += predicted_class
                            self.display_sentence += predicted_class
                    print(sentence)

            # display the translation if the user wants to
            if display_in_real_time:
                self._display_translation(frame_with_landmarks, self.display_sentence)
                window_created = True
                key = cv.waitKey(1) & 0xFF
                # Check if "q" key was pressed or the "Translation" window was closed and break the loop
                if key == ord("q") or self._is_translation_window_closed():
                    break

        self._close_video_translation(cap, display_in_real_time)
        sentence = sentence.capitalize()
        if self.text_processor:
            corrected_sentence = self.text_processor.correct_sentence(sentence)
            sentence = corrected_sentence if corrected_sentence else sentence
        return sentence

    def translate_image(self, image_path: str, model_name: str = "letters") -> str:
        """Translates the sign language found in the image in the given path.

        Args:
            image_path (str): The path of the image file to be translated.
            model_name (str, optional): The name of the model loaded. Defaults to 'letters'.

        Returns:
            str: The resulting translation.

        NOTE: This method only translates to letters given that no static signs mean words or concepts.
        """
        CONFIDENCE_THRESHOLD = 0.7
        self.processor = MediaPipeProcessor(self.mediapipe_confidence, "hands")
        prediction_model, prediction_model_signs = self.get_model(model_name)
        print(f"Processing image in: {image_path}")
        frame = cv.imread(image_path)
        if frame is None:
            print(f"Couldn't open image in: {image_path}")
            return
        resized_frame = cv.resize(frame, (640, 480))
        # process image and get the resulting landmarks
        results, _ = self.processor.process_image(resized_frame)
        if not self.processor.are_results_valid(results):
            print(
                f"No valid landmarks given the criteria of {self.processor.mode} model"
            )
            return
        # Extract the landmarks from both hands and save them in arrays
        keypoints = self.processor.keypoint_extraction(results)
        prediction = prediction_model.predict(keypoints[newaxis, :])
        if amax(prediction) > CONFIDENCE_THRESHOLD:
            predicted_class = prediction_model_signs[argmax(prediction)]
            return predicted_class
        else:
            return "The model is not confident enough about the translation."

    def _display_translation(self, frame: NDArray, translation: str):
        """Shows the current frame with the given translation.

        Args:
            frame (NDArray): The current opencv frame.
            translation (str): The curent translation.
        """
        cv_image = self._overwrite_frame_with_text(frame, translation)
        # Show the image on the display
        cv.imshow("Translation", cv_image)

    def _is_translation_window_closed(self) -> bool:
        """Returns wether if the translation window is closed or not.

        Returns:
            bool: True if the translation window is closed. False otherwise.
        """
        return cv.getWindowProperty("Translation", cv.WND_PROP_VISIBLE) < 1

    def _overwrite_frame_with_text(
        self, frame: NDArray, text: str = "", text_size: int = 40
    ) -> NDArray:
        """Overwrites the desired text in the given frame.

        Args:
            frame (NDArray): The opencv frame to overwrite.
            text (str): The desired text. Defaults to "". If no text is given self.display_sentence will be used instead.
            text_size (int, optional): The desired size in which the text will be written. Defaults to 40.

        Returns:
            NDArray: The overwritten frame.
        """
        if not text:
            text = self.display_sentence
        # Convert BGR to RGB for PIL
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pillow_image = Image.fromarray(rgb_frame)
        # convert the pillow Image to a drawable object
        draw_object = ImageDraw.Draw(pillow_image)
        font = ImageFont.truetype("arial.ttf", text_size)
        # prepare the text for display
        if self._is_text_overflowing(frame, text, draw_object, font):
            self._add_new_line_character()
            text = self.display_sentence
        text_X_coord, text_y_coord = self._get_translation_coordinates(
            frame, text, draw_object, font
        )
        draw_object.multiline_text(
            (text_X_coord, text_y_coord), text, (86, 24, 201), font, align="center"
        )
        # Convert PIL image (RGB) back to OpenCV image (BGR) and return it
        return cv.cvtColor(array(pillow_image), cv.COLOR_RGB2BGR)

    def _get_translation_x_coordinate(
        self,
        frame: NDArray,
        translation: str,
        draw_object: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
    ) -> int:
        """Uses the boundings of the given translation to calculate where it should be placed in the frame in the x axis (horizontally)
        to be centered.

        Args:
            frame (NDArray): The opencv frame.
            translation (str): The translation that will be written in the frame.
            draw_object (ImageDraw): The object that will write the translation.
            font (ImageFont): The font object that will be used to write the translation.

        Returns:
            int: The x axis value where the text should be to be centered.
        """
        # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
        bounding_box = draw_object.multiline_textbbox((0, 0), translation, font=font)
        text_width = bounding_box[2] - bounding_box[0]
        return (frame.shape[1] - text_width) // 2

    def _get_translation_y_coordinate(
        self,
        frame: NDArray,
        translation: str,
        draw_object: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
    ) -> int:
        """Uses the boundings of the given translation to calculate where it should be placed in the frame in the y axis (vertically).

        Args:
            frame (NDArray): The opencv frame.
            translation (str): The translation that will be written in the frame.
            draw_object (ImageDraw): The object that will write the translation.
            font (ImageFont): The font object that will be used to write the translation.

        Returns:
            int: The y axis value where the text should be.
        """
        # Calculate the size of the text to be displayed and the y coordinate for putting the text on the image
        bounding_box = draw_object.multiline_textbbox((0, 0), translation, font=font)
        text_height = bounding_box[3] - bounding_box[1]
        BOTTOM_MARGIN = 20
        return frame.shape[0] - text_height - BOTTOM_MARGIN

    def _get_translation_coordinates(
        self,
        frame: NDArray,
        translation: str,
        draw_object: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
    ) -> tuple[int, int]:
        """Uses the boundings of the given translation to calculate where it should be placed in the frame.

        Args:
            frame (NDArray): The opencv frame.
            translation (str): The translation that will be written in the frame.
            draw_object (ImageDraw): The object that will write the translation.
            font (ImageFont): The font object that will be used to write the translation.

        Returns:
            tuple: The (x,y) coordinates for the translation.
        """
        x_coordinate = self._get_translation_x_coordinate(
            frame, translation, draw_object, font
        )
        y_coordinate = self._get_translation_y_coordinate(
            frame, translation, draw_object, font
        )
        return x_coordinate, y_coordinate

    def _is_text_overflowing(
        self,
        frame: NDArray,
        text: str,
        draw_object: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
    ) -> bool:
        """Returns if the text is overflowing the frame.

        Args:
            frame (NDArray): The opencv frame.
            text (str): The text that will be written in the frame.
            draw_object (ImageDraw): The object that will write the text.
            font (ImageFont): The font object that will be used to write the text.

        Returns:
            bool: True if the text is overflowing the frame, False otherwise.
        """
        bounding_box = draw_object.textbbox((0, 0), text, font=font)
        text_width = bounding_box[2] - bounding_box[0]
        horizontal_space_left = frame.shape[1] - text_width
        return horizontal_space_left <= 140

    def _add_new_line_character(self):
        """Adds a new line character at the end of the display history's
        and display sentence's penultimate word. This method is only used
        for displaying purposes.
        """
        if self.lines_counter < 4:
            self.display_history[-2] = f"{self.display_history[-2]}\n"
            for i in range(len(self.display_sentence) - 1, 0, -1):
                if self.display_sentence[i] == " ":
                    self.display_sentence = (
                        self.display_sentence[:i]
                        + "\n"
                        + self.display_sentence[i + 1 :]
                    )
                    break
            self.lines_counter += 1
        else:
            self.display_history = [self.display_history[-1]]
            self.display_sentence = self.display_history[-1]

    def _close_video_translation(
        self, video_capture: cv.VideoCapture, display_in_real_time: bool
    ):
        """Closes given video capture and destroys all the opencv windows.

        Args:
            video_capture (cv.VideoCapture)
            display_in_real_time (bool): If the translation is being shown in real time or not.
        """
        # Release the video capture and close all windows
        if display_in_real_time:
            video_capture.release()
        cv.destroyAllWindows()
        # Shut off the server
        if self.text_processor and self.text_processor.language_tool:
            self.text_processor.language_tool.close()

    def load_model(
        self, signs: list[str], model_name: str, model_path: str = "models/model.keras"
    ):
        """Loads the model in the given path in the class' memory for later use.

        Args:
            signs (list[str]): A list containing the models known signs (could be words or letters).
            model_name (str): The name that will be given to the model in memory.
            model_path (str, optional): Where the model is. Defaults to "models/model.keras".
        """
        self.loaded_models[model_name] = (load_model(model_path), signs)

    def get_model(self, model_name: str) -> tuple[Model, list[str]]:
        """Returns the model stored in the class with the given name
        and its signs if found.

        Args:
            model_name (str): How the model was called when loaded.

        Raises:
            ValueError: Raised if no model with the given name is found.

        Returns:
            tuple[Model, list]: The found Model and its signs.
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name][0], self.loaded_models[model_name][1]
        else:
            raise ValueError(
                f"Model '{model_name}' isn't loaded. Load it first and try again."
            )
