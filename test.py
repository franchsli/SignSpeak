import os
from tools import *
import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer
import matplotlib.pyplot as plt
import tempfile

gesture_model_path: str = os.path.abspath("models/gesture_recognizer.task")
video_folder: str = os.path.abspath("datasets/10-words-slc-and-3-people/")
image_folder: str = os.path.abspath("datasets/Alphabet_SLC/")

# Set a custom temporary directory
created_models_dir = os.path.abspath("C:/Users/windows/Desktop/created_models")
os.makedirs(created_models_dir, exist_ok=True)
tempfile.tempdir = created_models_dir
print(f"Export directory: {tempfile.gettempdir()}")
print(f"Is directory writable? {os.access(tempfile.gettempdir(), os.W_OK)}")
assert os.access(tempfile.gettempdir(), os.W_OK)
# VideoGesturesRecognizer = VideoHandler(gesture_model_path, video_folder)
# VideoGesturesRecognizer.run()
# WebCamGestureRecognizer = WebCamHandler(gesture_model_path)
# WebCamGestureRecognizer.run()
#print(image_folder)
#labels = []
#for i in os.listdir(image_folder):
#    if os.path.isdir(os.path.join(image_folder, i)):
#        labels.append(i)
#print(labels)

# NUM_EXAMPLES = 5
# 
# for label in labels:
#     label_dir = os.path.join(image_folder, label)
#     example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
#     fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10, 2))
#     for i in range(NUM_EXAMPLES):
#         axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
#         axs[i].get_xaxis().set_visible(False)
#         axs[i].get_yaxis().set_visible(False)
#     fig.suptitle(f"Showing {NUM_EXAMPLES} examples for {label}")
# 
# plt.show()

data = gesture_recognizer.Dataset.from_folder(
    dirname=image_folder,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

hparams = gesture_recognizer.HParams(export_dir=os.path.abspath("created_models"))
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")

model.export_model()

