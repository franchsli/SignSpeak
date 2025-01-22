import os
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

gesture_model_path = os.path.abspath("models/gesture_recognizer.task")
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
mp_hands = mp.solutions.hands
VisionRunningMode = vision.RunningMode

video_folder = os.path.abspath("datasets/10-words-slc-and-3-people/")

# Create a gesture recognizer instance with the video mode:
gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=gesture_model_path),
    running_mode=VisionRunningMode.VIDEO,
)

global_timestamp = 0

with GestureRecognizer.create_from_options(
    gesture_options
) as recognizer, mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:

    for video_file in os.listdir(video_folder):
      video_path = os.path.join(video_folder, video_file)
      
      if not video_file.endswith((".mp4", ".avi", ".mov")):
          continue
      
      print(f"Processing video: {video_file}")
      cap = cv.VideoCapture(video_path)
      
      if not cap.isOpened():
          print(f"Failed to open video: {video_file}")
          continue
      
      frame_index = 0
      fps = cap.get(cv.CAP_PROP_FPS) or 30  # Default to 30 FPS if unknown
      
      while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Ensure global timestamp is used
        timestamp_ms = int(global_timestamp + (frame_index * 1000 / fps))
        result = recognizer.recognize_for_video(mp_image, timestamp_ms=timestamp_ms)
        
        # Hand landmarks detection
        landmarks_result = hands.process(rgb_frame)
        if landmarks_result and landmarks_result.multi_hand_landmarks:
            for hand_landmarks in landmarks_result.multi_hand_landmarks:
                # Draw landmarks with default MediaPipe styling
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )
        if result and result.gestures:
            print(f"Video: {video_file}, Frame {frame_index}: {result.gestures[0]}")
        else:
            print(f"Video: {video_file}, Frame {frame_index}: No gestures detected")
      
        cv.imshow("Video", frame)
      
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
      
        frame_index += 1
      
      # Update the global timestamp for the next video
      global_timestamp += int(cap.get(cv.CAP_PROP_FRAME_COUNT) * 1000 / fps)
      cap.release()

cv.destroyAllWindows()
