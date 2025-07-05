import os
from tools import VideoHandler

video_folder = os.path.abspath("data/ADIOS/")
video_folders_path = os.path.abspath("data")
video_folders = os.listdir(video_folders_path)
VideoGesturesRecognizer = VideoHandler("gesture_model_path", video_folder)
# VideoGesturesRecognizer.create_directories(os.path.abspath("test"))
#VideoGesturesRecognizer.run()
# create datasets code
"""
for video_folder in video_folders:
    VideoGesturesRecognizer.video_folder = os.path.join(
        video_folders_path, video_folder
    )
    VideoGesturesRecognizer.create_dataset(os.path.abspath("testing"))
"""
# training code (after all the datasets are created)
# VideoGesturesRecognizer.train(os.path.abspath("testing"))
