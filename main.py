"""File to test the trained model"""
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
from keras.models import load_model, Model
from language_tool_python import LanguageToolPublicAPI

# Set the path to the data directory
PATH = os.path.abspath('data')

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Load the trained model
model: Model = load_model('models/model.keras')

# Create an instance of the grammar correction tool
tool = LanguageToolPublicAPI('es')

# Initialize the lists
sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []
prediction_history = []
previous_sentence_length = 0
previous_sentence = []

# Access the camera and check if the camera is opened successfully
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(os.path.abspath("data/ADIOS/ADIOS1.mp4"))
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Run the loop while the camera is open
    while cap.isOpened():
        # Read a frame from the camera
        _, image = cap.read()
        resized_frame = cv.resize(image, (960, 540))
        # Process the image and obtain sign landmarks using image_process function from my_functions.py
        results, processed_image = image_process(resized_frame, holistic)
        # Draw the sign landmarks on the image using draw_landmarks function from my_functions.py
        frame_with_landmarks = draw_landmarks(processed_image, results)
        # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
        keypoints.append(keypoint_extraction(results))

        # Check if 10 frames have been accumulated
        if len(keypoints) == 10:
            # Convert keypoints list to a numpy array
            keypoints = np.array(keypoints)
            # Make a prediction on the keypoints using the loaded model
            prediction = model.predict(keypoints[np.newaxis, :, :])
            # Clear the keypoints list for the next set of frames
            keypoints = []

            # Only run grammar correction when sentence changes
            if len(sentence) > len(previous_sentence):
                grammar_result = tool.correct(' '.join(sentence))
                previous_sentence = sentence.copy()


            if np.amax(prediction) > 0.7:
                predicted_class = actions[np.argmax(prediction)]
                
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
        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

        # Check if the list is not empty
        if sentence:
            # Capitalize the first word of the sentence
            sentence[0] = sentence[0].capitalize()

        # Check if the sentence has at least two elements
        if len(sentence) >= 2:
            # Check if the last element of the sentence belongs to the alphabet (lower or upper cases)
            if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                # Check if the second last element of sentence belongs to the alphabet or is a new word
                if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                    # Combine last two elements
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(len(sentence) - 2)
                    sentence[-1] = sentence[-1].capitalize()


        if len(sentence) > previous_sentence_length:
            grammar_result = tool.correct(' '.join(sentence))
            previous_sentence_length = len(sentence)
            previous_sentence = sentence 
            print(sentence)
            print(grammar_result)

        if grammar_result:
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
            textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (frame_with_landmarks.shape[1] - textsize[0]) // 2

            # Draw the sentence on the frame_with_landmarks
            cv2.putText(frame_with_landmarks, grammar_result, (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 4, 45), 2, cv2.LINE_AA)
        else:
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the frame_with_landmarks
            textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (frame_with_landmarks.shape[1] - textsize[0]) // 2

            # Draw the sentence on the frame_with_landmarks
            cv2.putText(frame_with_landmarks, ' '.join(sentence), (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 4, 45), 2, cv2.LINE_AA)

        # Show the image on the display
        cv2.imshow('Camera', frame_with_landmarks)

        cv2.waitKey(1)

        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Shut off the server
    tool.close()
