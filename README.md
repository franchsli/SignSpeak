# SignSpeak

A sign language translator.

## TODO

***IMPORTANT***
**Read Claude Chat and rework:**

- Updates:
  - Create datasets for the keras model v2 and train the model.
- Performance:
  current import time = around 5 seconds.
  current translation time = around 6.3 times slower.
  - Motion-based filtering:
    Skip frames where hands move too fast (likely motion blur)
    Skip frames where hands are too close together (overlapping gestures)
    Skip frames during rapid hand position changes (transitioning between signs)

  - Confidence-based filtering:

    Skip frames where MediaPipe landmark confidence is below threshold
    Skip frames where hand detection confidence drops significantly from previous frame

  - Temporal filtering:

    Skip frames immediately after a prediction (cooldown period)
    Skip frames during the first/last second of video (setup/teardown gestures)

  - Spatial filtering:

    Skip frames where hands are too close to camera (too large/distorted)
    Skip frames where hands are partially out of frame bounds
    Skip frames where both hands are on same side of body (unnatural signing)

  - Stability filtering:

    Skip frames where landmark positions vary drastically from previous frame
    Skip frames during periods of high hand jitter/tremor

### Low priority

After completing the to-do list, check which one of the tasks in this list
stills needs a fix.

- Break down translate_video() into smaller functions (check Claude Chat).
- Implement UTF-8 encoding support in opencv putText()
- Check why running test.py takes so much time to start running (Ask Claude).
- Fully implement model testing into VideoHandler.
  - Try to run and understand main.py file.
  - Add needed_landmarks_present method to model testing.
  - Speed up main.py execution time (Ask Claude).
  - Implement something to check if the last_prediction is a
- Use VideoHandler functionalities into WebCamHandler.
- Implement an ImageHandler (think about it).
- Collect more data (sentences and letters) to train the VideoHandler with.

[use this repo](https://github.com/dgovor/Sign-Language-Translator)

## General Resources

[Main dataset used](https://www.kaggle.com/datasets/juanrrai/10-words-slc-and-3-people)

[Sign language processing](https://pypi.org/project/sign-language-tools/)

[Text to Sign language library docs](https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators)

[Datasets](https://www.kaggle.com/datasets?search=colombian+sign+language)

[More Data](https://www.youtube.com/watch?v=JMraBJsA9oI&list=PLI7rDimYXOdhyty-lEXsxQgiLfYKnnqmY&index=4)

[Very good dataset](https://bivl2ab.uis.edu.co/dataset-info)
