# SignSpeak

A sign language translator.

## TODO

## High priority

- Implement ImageHandler.

### Low priority

- Implement some of the suggestions from Claude and maybe convert the core folder classes to dataclasses.
- Add logging: Replace print statements.
- Remove "DEBUG" variable in translate_video() method.
- Collect more data (sentences and letters) to train the VideoHandler with.
- Implement frame division.
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
