import mediapipe as mp
import cv2 as cv
import numpy as np


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
