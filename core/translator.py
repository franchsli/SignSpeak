import cv2 as cv
from os import path
from numpy import array, amax, argmax, newaxis, ndarray
from PIL import ImageDraw, ImageFont, Image
from string import ascii_lowercase, ascii_uppercase
from keyboard import is_pressed
from predictor import GesturePredictor
from processor import MediaPipeProcessor
from text_processor import TextProcessor


class SignLanguageTranslator:
    def __init__(self, actions: ndarray, model_path: str = "models/model.keras", mediapipe_confidence: float = 0.75, language = "es"):
        """The generic class for a sign language translator

        Args:
            actions (ndarray): A numpy array containing the model's known words.
            model_path (str, optional): Where the keras model is.
            mediapipe_confidence (float, optional): The confidence of the mediapipe landmark detection. Defaults to 0.75.
            language (str, optional): The language's code that will be checked to correct
            the text. Defaults to "es".
        """
        
        self.processor = MediaPipeProcessor(mediapipe_confidence)
        self.predictor = GesturePredictor(actions, model_path)
        self.actions = actions
        self.text_processor = TextProcessor(language)
        

    def translate_video(self, video_input: str | int = 0, in_real_time: bool = True) -> str:
        """Translates the given video from CSL to spanish (currently).

        Args:
            video_input (str | int, optional): The path of the video file to be translated or 0 for the webcam. Defaults to 0.
            in_real_time (bool, optional): Whether if the translation should be displayed as it's translated or after the video is processed entirely.
            Defaults to True. Note that this variable will be True if the video_input is the webcam.
        
        Returns:
            str: The corrected translation if it was corrected successfully, the raw translation if not.
            
        NOTE: The real-time display shows uncorrected translation for immediate feedback.
                For grammatically corrected text, use the returned translation.
        """
        # set the parameter to true if the webcam is being translated
        in_real_time = True if video_input == 0 else in_real_time
        # set development variable to avoid unnecesary, excesive calls to the LanguageTool API
        DEBUG = True
        # Initialize the lists
        keypoints, last_prediction = [], []
        prediction_history, previous_prediction_history = [], []
        sentence = ""
        # Access the camera and check if the camera is opened successfully
        cap = cv.VideoCapture(video_input)
        if not cap.isOpened():
            print("Cannot access camera.")
            exit()

        # Create a holistic object for sign prediction
        with self.processor.holistic as holistic:
            # Run the loop while the camera is open
            while cap.isOpened():
                # Read a frame from the camera
                success, image = cap.read()
                if not success or image is None:
                    break
                resized_frame = cv.resize(image, (960, 540))
                # Process the image and obtain sign landmarks using image_process
                results, processed_image = self.processor.image_process(resized_frame, holistic)
                #print("Face:", results.face_landmarks.landmark)
                #print("Left hand:", results.left_hand_landmarks.landmark)
                #print("Pose", results.pose_landmarks.landmark)
                #print("Pose world", results.pose_world_landmarks.landmark)
                #print("Right hand:", results.right_hand_landmarks.landmark)
                if not self.processor.needed_landmarks_present(results):
                    print("Not enough landmarks present in the frame, skipping...")
                    continue
                if not self.processor.wrists_are_above_hips(results):
                    print("No wrist in the frame is above its closest hip, skipping...")
                    continue
                # Draw the sign landmarks on the image using draw_landmarks
                frame_with_landmarks = self.processor.draw_landmarks(processed_image, results)
                # Extract keypoints from the pose landmarks using keypoint_extraction
                keypoints.append(self.processor.keypoint_extraction(results))
                # Check if 10 frames have been accumulated
                if len(keypoints) == 10:
                    # Convert keypoints list to a numpy array
                    keypoints = array(keypoints)
                    # Make a prediction on the keypoints using the loaded model
                    prediction = self.predictor.model.predict(keypoints[newaxis, :, :])
                    # Clear the keypoints list for the next set of frames
                    keypoints = []
                    # Update variables when there's a new prediction
                    if len(prediction_history) > len(previous_prediction_history):
                        previous_prediction_history = prediction_history.copy()
                        sentence = " ".join(prediction_history)
                        print(sentence)
                    # Check if the maximum prediction value is above 0.7
                    if amax(prediction) > 0.7:
                        predicted_class = self.actions[argmax(prediction)]
                        
                        # Add prediction smoothing
                        prediction_history.append(predicted_class)
                        # Removes the oldest prediction
                        if len(prediction_history) > 3:
                            prediction_history.pop(0)
                        
                        # Use most common prediction in recent history
                        if prediction_history.count(predicted_class) >= 2:
                            if last_prediction != predicted_class:
                                sentence += f" {predicted_class}"
                                last_prediction = predicted_class
                # Limit the prediction_history length to 7 elements to make sure it fits on the screen
                # TODO: REWORK THIS LOGIC, TO IMPLEMENT MULTILINE TEXT (Check Pillow multiline_text)
                if len(prediction_history) > 7:
                    print("CUTTING THE prediction_history...")
                    prediction_history = prediction_history[-7:]
                # Reset if the "Spacebar" is pressed
                if is_pressed(" "):
                    sentence, keypoints, last_prediction,  = "", [], [],
                    prediction_history, previous_prediction_history = [], []
                    grammar_result = ""
                # Check if the prediction_history is not empty
                if prediction_history:
                    # ALERT: This is LETTER ONLY logic.
                    # Capitalize the first word of the prediction_history
                    prediction_history[0] = prediction_history[0].capitalize()
                # Check if the prediction_history has at least two elements and one of them is a letter (for letter correction)
                if len(prediction_history) >= 2 and len(prediction_history[-1]) == 1:
                    # check if the current letter predictions are valid
                    # ALERT: This is LETTER ONLY logic.
                    self._correct_current_letter_predictions()
                # display the translation if the user wants to
                if in_real_time:
                    self._display_translation(frame_with_landmarks, sentence)
                    # Check if the "Translation" window was closed and break the loop
                    if cv.getWindowProperty("Translation", cv.WND_PROP_VISIBLE) < 1:
                        break
            
            self._close_video_translation(cap, in_real_time)
            if self.text_processor.tool:
                corrected_sentence = self.text_processor.correct_sentence(sentence)
                sentence = corrected_sentence if corrected_sentence else sentence
            return sentence
        
    
    def _display_translation(self, frame: ndarray, translation: str):
        """Shows the current frame with the given translation.

        Args:
            frame (ndarray): The current opencv frame.
            translation (str): The curent translation.
        """
        cv_image = self._overwrite_frame_with_text(frame, translation)
        # Show the image on the display
        cv.imshow("Translation", cv_image)
        cv.waitKey(1)
    
    def _overwrite_frame_with_text(self, frame: ndarray, text: str, text_size: int = 40) -> ndarray:
        """Overwrites the desired text in the given frame.

        Args:
            frame (ndarray): The opencv frame to overwrite.
            text (str): The desired text.
            text_size (int, optional): The desired size in which the text will be written. Defaults to 40.

        Returns:
            ndarray: The overwritten frame.
        """
        # Convert BGR to RGB for PIL
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pillow_image = Image.fromarray(rgb_frame)
        # convert the pillow Image to a drawable object
        draw_object = ImageDraw.Draw(pillow_image)
        font = ImageFont.truetype("arial.ttf", text_size)
        text_X_coord = self._get_translation_x_coordinate(frame, text, draw_object, font)
        draw_object.text((text_X_coord, 470), text, (86, 24, 201), font)
        # Convert PIL image (RGB) back to OpenCV image (BGR) and return it
        return cv.cvtColor(array(pillow_image), cv.COLOR_RGB2BGR)
        
    
    def _get_translation_x_coordinate(self, frame: ndarray, translation: str, draw_object: ImageDraw, font: ImageFont) -> int:
        """Uses the boundings of the given translation to calculate where it should be placed in the frame in the x axis (horizontally)
        to be centered. 

        Args:
            frame (ndarray): The opencv frame.
            translation (str): The translation that will be written in the frame.
            draw_object (ImageDraw): The object that will write the translation.
            font (ImageFont): The font object that will be used to write the translation.

        Returns:
            int: The x axis value where the text should be to be centered.
        """
        # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
        bounding_box = draw_object.textbbox((0, 0), translation, font=font)
        text_width = bounding_box[2] - bounding_box[0]
        return (frame.shape[1] - text_width) // 2
    
    def _correct_current_letter_predictions(self):
        """Corrects the current letter predictions by checking if the predictions are letter and combine them.
        (e.g. ['H', 'O', 'L', 'A'] -> 'Hola').
        """
        global prediction_history
        # Check if the last element of the prediction_history belongs to the alphabet (lower or upper cases)
        if prediction_history[-1] in ascii_lowercase or prediction_history[-1] in ascii_uppercase:
            # Check if the penultimate element of prediction_history belongs to the alphabet or is a new word
            if prediction_history[-2] in ascii_lowercase or prediction_history[-2] in ascii_uppercase or (prediction_history[-2] not in self.actions and prediction_history[-2] not in list(x.capitalize() for x in self.actions)):
                # Combine last two elements
                prediction_history[-1] = prediction_history[-2] + prediction_history[-1]
                prediction_history.pop(len(prediction_history) - 2)
                prediction_history[-1] = prediction_history[-1].capitalize()
    
    def _close_video_translation(self, video_capture: cv.VideoCapture, in_real_time: bool):
        """Closes given video capture and destroys all the opencv windows.

        Args:
            video_capture (cv.VideoCapture)
            in_real_time (bool): If the translation is being shown in real time or not.
        """
        # Release the camera and close all windows
        if in_real_time:
            video_capture.release()
        cv.destroyAllWindows()
        # Shut off the server
        if self.text_processor.tool:
            self.text_processor.tool.close()