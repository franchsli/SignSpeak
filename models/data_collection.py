"""File to create the appropiate datasets."""

import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
import keyboard

video_folders_path = os.path.abspath("datasets/random_words/")
video_folders = os.listdir(video_folders_path)

# Define the actions (signs) that will be recorded and stored in the dataset
signs = np.array(video_folders)

# Define the number of sequences and frames to be recorded for each action
sequences = 30
frames = 10

# Set the path where the dataset will be stored
PATH = os.path.abspath("datasets/Alphabet_SLC/")

# Create directories for each action, sequence, and frame in the dataset
"""for action, sequence in product(signs, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass"""

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(
    os.path.abspath("datasets/random_words/BUENOS_DIAS/BUENOS_DIAS1.mp4")
)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
with mp.solutions.holistic.Holistic(
    min_detection_confidence=0.75, min_tracking_confidence=0.75
) as holistic:
    # Loop through each action, sequence, and frame to record data
    for action, sequence, frame in product(signs, range(sequences), range(frames)):
        # If it is the first frame of a sequence, wait for the spacebar key press to start recording
        if frame == 0:
            while True:
                if keyboard.is_pressed(" "):
                    break
                success, image = cap.read()
                if not success:
                    print("Error reading video or video ended")
                    break

                # Process image and get results
                results, processed_image = image_process(image, holistic)
                
                # Draw landmarks and display
                display_image = draw_landmarks(processed_image, results)

                # cv2.putText(image, 'Recording data for the "{}". Sequence number {}.'.format(action, sequence),
                #            (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                # cv2.putText(image, 'Pause.', (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                # cv2.putText(image, 'Press "Space" when you are ready.', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow("Camera", display_image)
                cv2.waitKey(1)

                # Check if the 'Camera' window was closed and break the loop
                if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
                    break
        else:
            # For subsequent frames, directly read the image from the camera
            success, image = cap.read()
            if not success:
                print("Error reading video or video ended")
                break
            # Process image and get results
            results, processed_image = image_process(image, holistic)
            
            # Draw landmarks and display
            display_image = draw_landmarks(processed_image, results)

            # Display text on the image indicating the action and sequence number being recorded
            # cv2.putText(image, 'Recroding data for the "{}". Sequence number {}.'.format(action, sequence),
            #            (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.imshow("Camera", display_image)
            cv2.waitKey(1)

        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
            break

        # Extract the landmarks from both hands and save them in arrays
        keypoints = keypoint_extraction(results)
        frame_path = os.path.join(PATH, action, str(sequence), str(frame))
        np.save(frame_path, keypoints)

    # Release the camera and close any remaining windows
    cap.release()
    cv2.destroyAllWindows()
