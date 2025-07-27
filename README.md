# SignSpeak

A sign language translator.

## TODO

## High priority

### Low priority

- Use tdqm to implement a load bar in dataset creation and other methods if needed (think about this).
  - **NOTE** The print() statements when turned into tqdm.write() make the bar look awful so maybe those need to be
  deleted and only the needed ones will stay.
- Add logging: Replace print statements. (or maybe not? or later?)
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
