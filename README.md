# SignSpeak

A sign language translator.

## TODO

## High priority

- Use MediaPipe hands in ImageHandler instead of holistic.
- Customize needed_landmarks_present() method in ImageHandler to match the context of images.

### Low priority

- Implement some of the suggestions from Claude and maybe convert the core folder classes to dataclasses.
- Add logging: Replace print statements.
- Use tdqm to implement a load bar  in dataset creation and other methods if needed.
- Remove "DEBUG" variable in translate_video() method.
- Collect more sentences to train the VideoHandler with.
- Collect letters to train the ImageHandler with.
- Implement frame division (think about this).
  Divide the frame by the number of people there and process the signs of everyone.
  - Add person detection (YOLO/MediaPipe Person Segmentation).
  - Crop bounding boxes around each person.
  - Run existing MediaPipe pipeline on each crop.
  - Merge results with person IDs.

[use this repo](https://github.com/dgovor/Sign-Language-Translator)

## General Resources

[Main dataset used](https://www.kaggle.com/datasets/juanrrai/10-words-slc-and-3-people)

[Sign language processing](https://pypi.org/project/sign-language-tools/)

[Text to Sign language library docs](https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators)

[Datasets](https://www.kaggle.com/datasets?search=colombian+sign+language)

[More Data](https://www.youtube.com/watch?v=JMraBJsA9oI&list=PLI7rDimYXOdhyty-lEXsxQgiLfYKnnqmY&index=4)

[Very good dataset](https://bivl2ab.uis.edu.co/dataset-info)

***IMPORTANT***

- Performance:
  current import time = around 5 seconds.
  current translation time = around 6.3 times slower.
