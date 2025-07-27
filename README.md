# SignSpeak

A sign language translator.

## TODO

## High priority

- Think about model switching in translations
  (whether if letters or words model should be used in the current video translation).
- Collect more sentences to train the VideoHandler with.
- Collect letters to train the ImageHandler with.

### Low priority

- Use tdqm to implement a load bar in dataset creation and other methods if needed (think about this).
  - **NOTE** The print() statements when turned into tqdm.write() make the bar look awful so maybe those need to be
  deleted and only the needed ones will stay.
- Add logging: Replace print statements. (or maybe not? or later?)
- Implement frame division (think about this).
  Divide the frame by the number of people there and process the signs of everyone.
  - Add person detection (YOLO/MediaPipe Person Segmentation).
  - Crop bounding boxes around each person.
  - Run existing MediaPipe pipeline on each crop.
  - Merge results with person IDs.

## General Resources

[Main repo used](https://github.com/dgovor/Sign-Language-Translator)

[Sign language processing](https://pypi.org/project/sign-language-tools/)

[Text to Sign language library docs](https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators)

[CSL in Youtube](https://www.youtube.com/watch?v=JMraBJsA9oI&list=PLI7rDimYXOdhyty-lEXsxQgiLfYKnnqmY&index=4)

## Datasets

[Main dataset used](https://www.kaggle.com/datasets/juanrrai/10-words-slc-and-3-people)

[Lexical Database of Colombian Sign Language](https://www.sign-lang.uni-hamburg.de/lr/compendium/lex/lesico.html)

[The dynamic Colombian sign language dataset for basic conversation LSC70](https://www.sciencedirect.com/science/article/pii/S2352340924011752)

[LSC50: Colombian Sign Language Video and Inertial Measurement dataset](https://www.nature.com/articles/s41597-024-04172-5)

[Kaggle datasets](https://www.kaggle.com/datasets?search=colombian+sign+language)

[Very good dataset](https://bivl2ab.uis.edu.co/dataset-info)

[ASL Dataset](https://how2sign.github.io/)

***IMPORTANT***

- Performance:
  current import time = around 5 seconds.
  current translation time = around 6.3 times slower.
- Think about this model naming:
  1. model's target prediction
  2. language
  3. "model"
  Where 1 is "words" or "letters", 2 is "CSL" or "ASL", etc...
