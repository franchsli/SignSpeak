import os
import cv2 as cv
from dataclasses import dataclass
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

@dataclass
class GestureHandler:
    model_path: str = None
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = vision.GestureRecognizer
    GestureRecognizerOptions = vision.GestureRecognizerOptions
    mp_hands = mp.solutions.hands
    VisionRunningMode = vision.RunningMode
    gesture_options = None

    
    def create_options(self) -> None:
        self.gesture_options = self.GestureRecognizerOptions(
        base_options=self.BaseOptions(model_asset_path=self.model_path),
        running_mode=self.VisionRunningMode.VIDEO,
        )



@dataclass
class VideoHandler(GestureHandler):
    video_folder: str = None
    global_timestamp: int = 0
    

    def handle_gesture(self, video_file, frame_index: int, result) -> None:
        if result.gestures[0][0].category_name != 'None':
            print(f"Video: {video_file}, Frame {frame_index}: {result.gestures[0][0].category_name}")
        else:
            print(f"Video: {video_file}, Frame {frame_index}: Gesture does not belongs to a category.")


    def run(self) -> None:
        self.create_options()
        with self.GestureRecognizer.create_from_options(
                self.gesture_options
            ) as recognizer, self.mp_hands.Hands(
                model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as hands:

            for video_file in os.listdir(self.video_folder):
                video_path: str = os.path.join(self.video_folder, video_file)
            
                if not video_file.endswith((".mp4", ".avi", ".mov")):
                    continue
            
                print(f"Processing video: {video_file}")
                cap = cv.VideoCapture(video_path)
            
                if not cap.isOpened():
                    print(f"Failed to open video: {video_file}")
                    continue
            
                frame_index: int = 0
                fps = cap.get(cv.CAP_PROP_FPS) or 30  # Default to 30 FPS if unknown
            
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    # Ensure global timestamp is used
                    timestamp_ms = int(self.global_timestamp + (frame_index * 1000 / fps))
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
                        self.handle_gesture(video_file, frame_index, result)

                    else:
                        print(f"Video: {video_file}, Frame {frame_index}: No gestures detected")
                
                    cv.imshow("Video", frame)
                
                    if cv.waitKey(1) & 0xFF == ord("q"):
                        self.stop()
                        break
                
                    frame_index += 1
                
                # Update the global timestamp for the next video
                self.global_timestamp += int(cap.get(cv.CAP_PROP_FRAME_COUNT) * 1000 / fps)
                cap.release()

    def stop(self) -> None:
        cv.destroyAllWindows()

    
