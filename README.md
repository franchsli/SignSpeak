# SignSpeak

A sign language translator.

## TODO

***IMPORTANT***
**Read Claude Chat and rework:**

- File Organization.
- Model Training Logic (tools.py).
  - Fix with sliding window.
- Real-time Prediction Issues (main.py).
  - Grammar correction runs every frame (expensive).
  - Confidence threshold (0.9) might be too high.
  - Letter combination logic is complex.
- Resource Management.
- Architecture (classes).

### Low priority

After completing the to-do list, check which one of the tasks in this list
stills needs a fix.

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

[Sign language processing](https://pypi.org/project/sign-language-tools/)

[Text to Sign language library docs](https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators)

[Datasets](https://www.kaggle.com/datasets?search=colombian+sign+language)

[More Data](https://www.youtube.com/watch?v=JMraBJsA9oI&list=PLI7rDimYXOdhyty-lEXsxQgiLfYKnnqmY&index=4)

[Very good dataset](https://bivl2ab.uis.edu.co/dataset-info)
