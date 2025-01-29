import os
from tools import *

video_folder: str = os.path.abspath("datasets/10-words-slc-and-3-people/")
image_folder: str = os.path.abspath("datasets/Alphabet_SLC/")
created_models_dir = os.path.abspath("created_models")

# VideoGesturesRecognizer = VideoHandler(gesture_model_path, video_folder)
# VideoGesturesRecognizer.run()
# WebCamGestureRecognizer = WebCamHandler(gesture_model_path)
# WebCamGestureRecognizer.run()