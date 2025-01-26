# SignSpeak

A sign language translator.

## TODO

- Increase VideoHandler and WebCamHandler fps.
- Train the model to recognize sign language.

## General Resources

[More mediapipe docs](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)

[Custom gesture recognition](https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer)

[Datasets](https://www.kaggle.com/datasets?search=colombian+sign+language)

[More Data](https://www.youtube.com/watch?v=JMraBJsA9oI&list=PLI7rDimYXOdhyty-lEXsxQgiLfYKnnqmY&index=4)

[Very good dataset](https://bivl2ab.uis.edu.co/dataset-info)

[More models](https://huggingface.co/models?sort=trending)

[Useful thing](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker)

## Problem Resources

[If problem with mediapipe](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-microsoft-visual-c-redistributable-version)

[Problem with mediapipe model maker](https://github.com/google-ai-edge/mediapipe/issues/4975)

[Problem with mediapipe model maker](https://github.com/google-ai-edge/mediapipe/issues/5214)

[Problem with tensorflow text](https://github.com/tensorflow/text?tab=readme-ov-file#install-using-pip)

## How to fix

If encountering problems with mediapipe model maker because tensorflow-text is not installed (Windows)
delete the text folder in mediapipe model maker package manually and delete the import statement in the init.py file
that imports the text folder.
