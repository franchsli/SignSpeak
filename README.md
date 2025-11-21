# SignSpeak

A sign language translator.

## TODO

## High priority

- Test everything.
- Write tests.
- Update readme (add project purpose, functionality, etc).
  - (maybe) separate tasks list in a separate file.
- Think about model switching in translations
  (whether if letters or words model should be used in the current video translation).

### Low priority

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

Your data folder (where you store videos or images to train the models) should look like this:

``` ASCII
.
└── data/
    ├── LABEL/
    │   ├── LABEL1
    │   ├── LABEL2
    │   ├── LABEL3
    │   └── ...
    ├── ANOTHER_LABEL/
    │   ├── ANOTHER_LABEL1
    │   ├── ANOTHER_LABEL2
    │   ├── ANOTHER_LABEL3
    │   └── ...
    └── ...
```

Where "LABEL" is the uppercase word or letter corresponding to the signs showed in the files.
The files must be named after the label AND a unique number inside the folder.
The dataset creation depends on this structure.

After using your data to create a dataset (via VideHandler), the resulting dataset would look like this:

``` ASCII
.
└── dataset/
    ├── LABEL/
    │   ├── 1_frame_1.npy
    │   ├── 1_frame_15.npy
    │   ├── ...
    │   ├── 2_frame_73.npy
    │   └── ...
    └── ...
```

It uses the same labels inside your data folder to name the folders there and
obviously adds the numpy data from the files in data/.
The numpy files are named this way (if the dataset is created with VideoHandler):
  {source_file_index}frame{source_frame_index}.npy
  (with underscores between the "variables")
This is called like this for debugging purposes, as you can know which is the first usable frame in a video.
If the dataset is created with ImageHandler dataset would look like this:

``` ASCII
.
└── dataset/
    ├── LABEL/
    │   ├── 1.npy
    │   ├── 2.npy
    │   └── ...
    ├── ANOTHER_LABEL/
    │   ├── 1.npy
    │   ├── 2.npy
    │   └── ...
    └── ...
```
