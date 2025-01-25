import os
from tools import *
import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer

import matplotlib.pyplot as plt

gesture_model_path: str = os.path.abspath("models/gesture_recognizer.task")
video_folder: str = os.path.abspath("datasets/10-words-slc-and-3-people/")
image_folder: str = os.path.abspath("datasets/Alphabet_SLC/")

#VideoGesturesRecognizer = VideoHandler(gesture_model_path, video_folder)
#VideoGesturesRecognizer.run()
#WebCamGestureRecognizer = WebCamHandler(gesture_model_path)
#WebCamGestureRecognizer.run()
print(image_folder)
labels = []
for i in os.listdir(image_folder):
  if os.path.isdir(os.path.join(image_folder, i)):
    labels.append(i)
print(labels)

NUM_EXAMPLES = 5

for label in labels:
  label_dir = os.path.join(image_folder, label)
  example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
  fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10,2))
  for i in range(NUM_EXAMPLES):
    axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
  fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {label}')

plt.show()