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

If you want to use the [language tool](https://languagetool.org/)
[text correction](docs/TEXT_CORRECTION.md) to correct your translations'
grammatical errors and such, you'll need to have Java 17 (or any version above it) installed.

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

                        pytest .\testing\tests.py

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
with SignLanguageTranslator() as translator:
    # Create an array of sign labels by listing the contents of the data directory
    signs = listdir("data")
    # load the models down here
    translator.load_model(signs, "model_name", "models/model_path.keras")
    translation = translator.translate_video("some_video.mp4")
    print(translation)
```

Using the "with" statement is adviced so the translator closes automatically
after you're done with it, but you can close it manually too:

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
print(translation)
translator.close()
```

Don't have anything? See the complete guide:

1. [Dataset creation.](docs/DATASET_CREATION.md)
2. [Model training.](docs/MODEL_TRAINING.MD)
3. [Translation.](docs/TRANSLATION.MD)
4. [Text correction.](docs/TEXT_CORRECTION.md)

## General Resources

These are resources I looked at while developing this project:

- [Main repo used](https://github.com/dgovor/Sign-Language-Translator), this is where I learnt the basics for this project.

- [Text to Sign language library docs](https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators)

- [CSL in Youtube](https://www.youtube.com/watch?v=JMraBJsA9oI&list=PLI7rDimYXOdhyty-lEXsxQgiLfYKnnqmY&index=4)

## Datasets

These are the datasets I found while doing research:

- [Main dataset used](https://www.kaggle.com/datasets/juanrrai/10-words-slc-and-3-people)
This is the dataset I used for testing while developing this project.

- [Lexical Database of Colombian Sign Language](https://www.sign-lang.uni-hamburg.de/lr/compendium/lex/lesico.html)

- [The dynamic Colombian sign language dataset for basic conversation LSC70](https://www.sciencedirect.com/science/article/pii/S2352340924011752)

- [LSC50: Colombian Sign Language Video and Inertial Measurement dataset](https://www.nature.com/articles/s41597-024-04172-5)

- [Kaggle datasets](https://www.kaggle.com/datasets?search=colombian+sign+language)

- [Very good dataset](https://bivl2ab.uis.edu.co/dataset-info)

- [ASL Dataset](https://how2sign.github.io/)

You're free to try any of these at your own risks, I mean, I only tested the first one, the others should be fine but I'll not
take responsability.

## Performance

Tested on Windows 10 with Intel(R) Celeron(R) CPU G1610 @ 2.60GHz.

- Current import time = around 5 seconds.
- Current video translation time = around 6.3 times slower.
around 4 times slower if it isn't displayed and uses "hands" mode.
(this is not viable)
