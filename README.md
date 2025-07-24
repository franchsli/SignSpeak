# SignSpeak

A sign language translator.

## TODO

## High priority

- Implement translate_image in SignLanguageTranlator.
- Add logging: Replace print statements.

### Low priority

- Use tdqm to implement a load bar  in dataset creation and other methods if needed.
- Remove "DEBUG" variable in translate_video() method.
- Set model_path to "models/word_model.keras in VideoHandler and "models/letter_model.keras"
  in ImageHandler (think about this).
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

[ASL Dataset](https://how2sign.github.io/)

***IMPORTANT***

- Performance:
  current import time = around 5 seconds.
  current translation time = around 6.3 times slower.
