import os
from tools import *

gesture_model_path: str = os.path.abspath("models/gesture_recognizer.task")
video_folder: str = os.path.abspath("datasets/10-words-slc-and-3-people/")

#VideoGesturesRecognizer = VideoHandler(gesture_model_path, video_folder)
#VideoGesturesRecognizer.run()
WebCamGestureRecognizer = WebCamHandler(gesture_model_path)
WebCamGestureRecognizer.run()
