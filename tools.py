import os
import cv2 as cv
from dataclasses import dataclass


@dataclass
class GestureHandler:
    model_path: str = None


@dataclass
class VideoHandler(GestureHandler):
    video_folder: str = None
    global_timestamp: int = 0

    def run(self) -> None:

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
                success, frame = cap.read()
                if not success:
                    break
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


class WebCamHandler(GestureHandler):
    def run(self) -> None:
        self.create_options(self.VisionRunningMode.LIVE_STREAM, self.handle_gesture)
        cap = cv.VideoCapture(0)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Add the image dimensions to the gesture recognizer options
        self.gesture_options.base_options.frame_width = frame_width
        self.gesture_options.base_options.frame_height = frame_height

        while cap.isOpened():
            frame_index: int = 0
            fps = cap.get(cv.CAP_PROP_FPS) or 30  # Default to 30 FPS if unknown

            while True:
                success, frame = cap.read()
                if not success:
                    continue

                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # Ensure global timestamp is used
                timestamp_ms = int(frame_index * 1000 / fps)

                cv.imshow("WebCam", frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    self.stop()
                    break

                frame_index += 1

            cap.release()

    def stop(self) -> None:
        cv.destroyAllWindows()
