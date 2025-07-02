import cv2 as cv
from os import path, listdir
from numpy import array, amax, argmax, newaxis, ndarray
from string import ascii_lowercase, ascii_uppercase
from time import time
from keyboard import is_pressed
from predictor import GesturePredictor
from processor import MediaPipeProcessor
from text_processor import TextProcessor


class SignLanguageTranslator:
    def __init__(self, model_path: str, actions: ndarray, mediapipe_confidence: float = 0.75, language = "es"):
        """The generic class for a sing language translator

        Args:
            model_path (str): Where the keras model is.
            actions (ndarray): A numpy array containing the model's known words.
            mediapipe_confidence (float, optional): The confidence of the mediapipe landmark detection. Defaults to 0.75.
            language (str, optional): The language's code that will be checked to correct
            the text. Defaults to "es".
        """
        
        self.processor = MediaPipeProcessor(mediapipe_confidence)
        self.predictor = GesturePredictor(model_path, actions)
        self.actions = actions
        self.text_processor = TextProcessor(language)
        

    def translate_video(self, video_input: str | int = 0, in_real_time: bool = True):
        """Translates the given video from CSL to spanish (currently).

        Args:
            video_input (str | int, optional): The path of the video file to be translated or 0 for the webcam. Defaults to 0.
            in_real_time (bool, optional): Whether if the translation should be displayed as it's translated or after the video is processed entirely.
            Note that this variable will be True if the video_input is the webcam.
        """
        start = time()
        # set the parameter to true if the webcam is being translated
        in_real_time = True if video_input == 0 else in_real_time
        # set development variable to avoid unnecesary, excesive calls to the LanguageTool API
        DEBUG = True
        # Initialize the lists
        sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []
        prediction_history = []
        previous_sentence = []
        # Access the camera and check if the camera is opened successfully
        # cap = cv.VideoCapture(0)
        cap = cv.VideoCapture(path.abspath("data/ADIOS/ADIOS1.mp4"))
        if not cap.isOpened():
            print("Cannot access camera.")
            exit()

        # Create a holistic object for sign prediction
        with self.processor.holistic as holistic:
            # Run the loop while the camera is open
            try:
                while cap.isOpened():
                    # Read a frame from the camera
                    _, image = cap.read()
                    resized_frame = cv.resize(image, (960, 540))
                    # Process the image and obtain sign landmarks using image_process function from my_functions.py
                    results, processed_image = self.processor.image_process(resized_frame, holistic)
                    # Draw the sign landmarks on the image using draw_landmarks function from my_functions.py
                    frame_with_landmarks = self.processor.draw_landmarks(processed_image, results)
                    # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
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
                        if len(sentence) > len(previous_sentence):
                            if self.text_processor.tool and not DEBUG:
                                grammar_result = self.text_processor.correct_sentence(' '.join(sentence))
                            else:
                                grammar_result = None
                            previous_sentence = sentence.copy()
                            print(sentence)
                            print(grammar_result if grammar_result else None)

                        # Check if the maximum prediction value is above 0.7
                        if amax(prediction) > 0.7:
                            predicted_class = self.actions[argmax(prediction)]
                            
                            # Add prediction smoothing
                            prediction_history.append(predicted_class)
                            if len(prediction_history) > 3:
                                prediction_history.pop(0)
                            
                            # Use most common prediction in recent history
                            if prediction_history.count(predicted_class) >= 2:
                                if last_prediction != predicted_class:
                                    sentence.append(predicted_class)
                                    last_prediction = predicted_class

                    # Limit the sentence length to 7 elements to make sure it fits on the screen
                    if len(sentence) > 7:
                        sentence = sentence[-7:]

                    # Reset if the "Spacebar" is pressed
                    if is_pressed(' '):
                        sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

                    # Check if the list is not empty
                    if sentence:
                        # Capitalize the first word of the sentence
                        sentence[0] = sentence[0].capitalize()

                    # Check if the sentence has at least two elements
                    if len(sentence) >= 2:
                        # Check if the last element of the sentence belongs to the alphabet (lower or upper cases)
                        if sentence[-1] in ascii_lowercase or sentence[-1] in ascii_uppercase:
                            # Check if the second last element of sentence belongs to the alphabet or is a new word
                            if sentence[-2] in ascii_lowercase or sentence[-2] in ascii_uppercase or (sentence[-2] not in self.actions and sentence[-2] not in list(x.capitalize() for x in self.actions)):
                                # Combine last two elements
                                sentence[-1] = sentence[-2] + sentence[-1]
                                sentence.pop(len(sentence) - 2)
                                sentence[-1] = sentence[-1].capitalize()

                    if grammar_result:
                        self.display_translation(frame_with_landmarks, grammar_result)
                    else:
                        self.display_translation(frame_with_landmarks, ' '.join(sentence))

                    # Show the image on the display
                    cv.imshow('Camera', frame_with_landmarks)

                    cv.waitKey(1)

                    # Check if the 'Camera' window was closed and break the loop
                    if cv.getWindowProperty('Camera',cv.WND_PROP_VISIBLE) < 1:
                        break
            
            except KeyboardInterrupt:
                pass

            finally:
                self._close_video_translation(cap)
                end = time()
                print(f"Proccessing 6 seconds long video took {end - start} seconds")
    
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