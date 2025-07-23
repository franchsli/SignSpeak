import numpy as np
import cv2 as cv
import mediapipe as mp
from enum import StrEnum
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.hands import Hands

class ProcessorMode(StrEnum):
    HANDS = "hands"
    HOLISTIC = "holistic"


class MediaPipeProcessor:
    def __init__(self, confidence: float = 0.75, mode: str = "holistic"):
        self.mode = mode
        if self.mode == ProcessorMode.HOLISTIC:
            self.model = Holistic(
                min_detection_confidence=confidence,
                min_tracking_confidence=confidence
            )
        elif self.mode == ProcessorMode.HANDS:
            self.model = Hands(
                static_image_mode=True,
                min_detection_confidence=confidence,
                min_tracking_confidence=confidence
            )

    def needed_landmarks_present(self, results) -> bool:
        """Returns True if the needed landmarks in the class mode are present. In the 'holistic' mode it means that the pose landmarks
        and at least one hand's landmarks are present. In the 'hands' mode, it means that at least one hand is present, returns
        False otherwise.

        Args:
            results: Either the holistic or hands model landmarks processing results.

        Returns:
            bool: If a pose and at least a hand are present (holistic mode) or at least a hand is present (hands mode).
        """
        if self.mode == ProcessorMode.HOLISTIC:
            pose = results.pose_landmarks
            left_hand = results.left_hand_landmarks
            right_hand = results.right_hand_landmarks
            return (pose and left_hand) or (pose and right_hand)
        
        elif self.mode == ProcessorMode.HANDS:
            return results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
    
    def wrists_are_above_hips(self, results) -> bool:
        """Returns True if at least one wrist is above
        its closest hip, False otherwise.

        Args:
            results: The holistic landmarker results.

        Returns:
            bool: If at least one wrist is above its closest hip.
        """
        mp_holistic = mp.solutions.holistic
        pose = results.pose_landmarks.landmark
        left_hip, left_wrist = pose[mp_holistic.PoseLandmark.LEFT_HIP].y, pose[mp_holistic.PoseLandmark.LEFT_WRIST].y
        right_hip, right_wrist = pose[mp_holistic.PoseLandmark.RIGHT_HIP].y, pose[mp_holistic.PoseLandmark.RIGHT_WRIST].y
        return (left_hip > 0.1 + left_wrist) or (right_hip > 0.1 + right_wrist)


    def draw_landmarks(self, image: np.ndarray, results) -> np.ndarray:
        """
        Draw the landmarks on the image.

        Args:
            image (numpy.ndarray): The input image.
            results: The landmarks detected by Mediapipe (The holistic landmarker results).

        Returns:
            numpy.ndarray: The image with drawn landmarks
        """
        # Make a copy of the image to ensure it's writable
        image = image.copy()

        if self.mode == ProcessorMode.HOLISTIC:
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
        
        elif self.mode == ProcessorMode.HANDS:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        return image

    def image_process(self, image: np.ndarray):
        """
        Process the image and obtain sign landmarks.

        Args:
            image (numpy.ndarray): The input image.

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
        # TODO: Improve this, model processing, it's so slow that it adds atleast 29 seconds on a 6 seconds long video (and 30 on average)
        results = self.model.process(image_rgb)
        image_rgb.flags.writeable = True

        # Convert back to BGR for OpenCV operations
        processed_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)

        return results, processed_image

    def keypoint_extraction(self, results) -> np.ndarray:
        """
        Extract the keypoints from the sign landmarks.

        Args:
            results: The processed results containing sign landmarks (The holistic landmarker results).

        Returns:
            numpy.ndarray: The extracted keypoints.
        """
        if self.mode == ProcessorMode.HOLISTIC:
            return self._extract_holistic_keypoints(results)
        
        elif self.mode == ProcessorMode.HANDS:
            raise NotImplementedError("NEED TO CODE A FUNCTION TO RETURN THE HANDS-ONLY KEYPOINTS")
    
    def _extract_holistic_keypoints(self, results) -> np.ndarray:
        # Extract the keypoints for the left hand if present, otherwise set to zeros
        left_hand = (
            np.array(
                [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
            ).flatten()
            if results.left_hand_landmarks
            else np.zeros(63)
        )
        # Extract the keypoints for the right hand if present, otherwise set to zeros
        right_hand = (
            np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            ).flatten()
            if results.right_hand_landmarks
            else np.zeros(63)
        )
        # Concatenate the keypoints for both hands
        return np.concatenate([left_hand, right_hand])