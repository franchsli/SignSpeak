import os
from tools import *
video_folder: str = os.path.abspath("datasets/10-words-slc-and-3-people/")
image_folder: str = os.path.abspath("datasets/Alphabet_SLC/")
created_models_dir = os.path.abspath("created_models")

# VideoGesturesRecognizer = VideoHandler(gesture_model_path, video_folder)
# VideoGesturesRecognizer.run()
# WebCamGestureRecognizer = WebCamHandler(gesture_model_path)
# WebCamGestureRecognizer.run()

video_folder = os.path.abspath("data/ADIOS/")
video_folders_path = os.path.abspath("data")
video_folders = os.listdir(video_folders_path)
VideoGesturesRecognizer = VideoHandler("gesture_model_path", video_folder)
VideoGesturesRecognizer.create_directories(os.path.abspath("test"))
"""for video_folder in video_folders:
    VideoGesturesRecognizer.video_folder = os.path.abspath(
        f"datasets/random_words/{video_folder}/"
    )
    #VideoGesturesRecognizer.create_dataset(image_folder)
    VideoGesturesRecognizer.train(os.path.abspath("test"))
"""