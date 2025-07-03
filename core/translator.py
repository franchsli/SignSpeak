import cv2 as cv
from os import path
from numpy import array, amax, argmax, newaxis, ndarray
from string import ascii_lowercase, ascii_uppercase
from keyboard import is_pressed
from predictor import GesturePredictor
from processor import MediaPipeProcessor
from text_processor import TextProcessor


class SignLanguageTranslator:
    def __init__(self, actions: ndarray, model_path: str = "models/model.keras", mediapipe_confidence: float = 0.75, language = "es"):
        """The generic class for a sing language translator

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
        

    def translate_video(self, video_input: str | int = 0, in_real_time: bool = True):
        """Translates the given video from CSL to spanish (currently).

        Args:
            video_input (str | int, optional): The path of the video file to be translated or 0 for the webcam. Defaults to 0.
            in_real_time (bool, optional): Whether if the translation should be displayed as it's translated or after the video is processed entirely.
            Note that this variable will be True if the video_input is the webcam.
        """
        # set the parameter to true if the webcam is being translated
        in_real_time = True if video_input == 0 else in_real_time
        # set development variable to avoid unnecesary, excesive calls to the LanguageTool API
        DEBUG = True
        # Initialize the lists
        keypoints, last_prediction = [], []
        prediction_history = []
        sentence, grammar_result = "", ""
        # Access the camera and check if the camera is opened successfully
        # cap = cv.VideoCapture(0)
        cap = cv.VideoCapture(path.abspath("data/ADIOS/ADIOS1.mp4"))
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
                    # Only run grammar correction when sentence changes
                    if len(prediction_history) > len(previous_prediction_history):
                        if self.text_processor.tool and not DEBUG:
                            grammar_result = self.text_processor.correct_sentence(' '.join(sentence))
                        else:
                            grammar_result = None
                        previous_prediction_history = prediction_history.copy()
                        sentence = " ".join(prediction_history)
                        print(sentence)
                        print(grammar_result if grammar_result else None)
                    # Check if the maximum prediction value is above 0.7
                    if amax(prediction) > 0.7:
                        predicted_class = self.actions[argmax(prediction)]
                        
                        # Add prediction smoothing
                        prediction_history.append(predicted_class)
                        # What?
                        if len(prediction_history) > 3:
                            prediction_history.pop(0)
                        
                        # Use most common prediction in recent history
                        if prediction_history.count(predicted_class) >= 2:
                            if last_prediction != predicted_class:
                                sentence += f" {predicted_class}"
                                last_prediction = predicted_class
                # Limit the sentence length to 7 elements to make sure it fits on the screen
                if len(sentence) > 7:
                    sentence = sentence[-7:]
                # Reset if the "Spacebar" is pressed
                if is_pressed(' '):
                    sentence, keypoints, last_prediction,  = "", [], [],
                # Check if the list is not empty
                if prediction_history:
                    # Capitalize the first word of the prediction_history
                    prediction_history[0] = prediction_history[0].capitalize()
                # Check if the prediction_history has at least two elements
                if len(prediction_history) >= 2:
                    # Check if the last element of the prediction_history belongs to the alphabet (lower or upper cases)
                    if prediction_history[-1] in ascii_lowercase or prediction_history[-1] in ascii_uppercase:
                        # Check if the second last element of prediction_history belongs to the alphabet or is a new word
                        if prediction_history[-2] in ascii_lowercase or prediction_history[-2] in ascii_uppercase or (prediction_history[-2] not in self.actions and prediction_history[-2] not in list(x.capitalize() for x in self.actions)):
                            # Combine last two elements
                            prediction_history[-1] = prediction_history[-2] + prediction_history[-1]
                            prediction_history.pop(len(prediction_history) - 2)
                            prediction_history[-1] = prediction_history[-1].capitalize()
                # display the translation if the user wants to
                if in_real_time:
                    if grammar_result:
                        self.display_translation(frame_with_landmarks, grammar_result)
                    else:
                        self.display_translation(frame_with_landmarks, sentence)
                    # Show the image on the display
                    cv.imshow('Camera', frame_with_landmarks)
                cv.waitKey(1)
                # Check if the 'Camera' window was closed and break the loop
                if cv.getWindowProperty('Camera',cv.WND_PROP_VISIBLE) < 1:
                    break
        
            self._close_video_translation(cap)
            return sentence if not in_real_time else None

    
    def display_translation(self, frame: ndarray, translation: str):
        # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
        textsize = cv.getTextSize(translation, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_X_coord = (frame.shape[1] - textsize[0]) // 2
        # Draw the sentence on the frame
        cv.putText(frame, translation, (text_X_coord, 470),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (210, 4, 45), 2, cv.LINE_AA)
    
    def _close_video_translation(self, video_capture: cv.VideoCapture):
        # Release the camera and close all windows
        video_capture.release()
        cv.destroyAllWindows()
        # Shut off the server
        if self.text_processor.tool:
            self.text_processor.tool.close()