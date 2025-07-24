import os
from tools import VideoHandler, ImageHandler

video_folder = os.path.abspath("data/ADIOS/")
video_folders_path = os.path.abspath("data")
video_folders = os.listdir(video_folders_path)
#VideoGesturesRecognizer = VideoHandler(data_parent_folder=video_folders_path)
# VideoGesturesRecognizer.create_directories(os.path.abspath("test"))
# create datasets code
# VideoGesturesRecognizer.create_dataset(os.path.abspath("testing"))

# training code (after all the datasets are created)
# VideoGesturesRecognizer.train(os.path.abspath("test"))

ImageSignHandler = ImageHandler(data_parent_folder=os.path.abspath("letters_data"))
ImageSignHandler.create_dataset(os.path.abspath("letters_test"))
ImageSignHandler.train(os.path.abspath("letters_test"), "models/letters.keras")