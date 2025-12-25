# SignSpeak

## Overview

A sign language translator model maker and runner.
With SignSpeak, you can train models to translate your desired language
and then test your model!

## Requirements

Before setting this up, you **MUST** have python **3.11**
installed, newer versions don't work and previous versions weren't tested.
[uv](https://docs.astral.sh/uv/) setup (pyproject.toml, lock file, etc)
will be added in newer versions, the project will use pip as for now.

## Quickstart

1.- Clone this repository:

                        git clone https://github.com/franchsli/SignSpeak

2.- Create a virtual environment:

                        python -m venv venv

3.- Activate virtual environment.

                        .\venv\Scripts\activate

4.- Install dependencies:

                        pip install -r requirements.txt --no-deps
**NOTE**:  `--no-deps` flags is required because it'll install non tested libraries
and you may encounter problems in the future.

5.- Ensure everything works:

                        pytest tests.py

If all tests passed, you're good to go!
It'll raise `DeprecationWarning: module 'sre_constants' is deprecated`
but don't worry, this happens because tensorflow is installing that deprecated module.

## Basic usage

After you have a model, you can use it to translate any media, like this:

``` python3
"""Translating a video file"""
# main.py in root
from os import listdir
from core.translator import SignLanguageTranslator
# Create an array of sign labels by listing the contents of the data directory
signs = listdir("data")
translator = SignLanguageTranslator()
# load the models down here
translator.load_model(signs, "model_name", "models/model_path.keras")
translation = translator.translate_video("some_video.mp4")
```

Don't have anything? See the complete guide:

1. [Dataset creation.](docs/DATASET_CREATION.md)
2. [Model training.](docs/MODEL_TRAINING.MD)
3. [Translation.](docs/TRANSLATION.MD)

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
  current video translation time = around 6.3 times slower.
  around 4 times slower if it isn't displayed and uses "hands" mode.
  (this is not viable)
