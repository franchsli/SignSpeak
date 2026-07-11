# SignSpeak

## Overview

A sign language translator model maker and runner.
With SignSpeak, you can train models to translate your desired language
and then test your model!

https://github.com/user-attachments/assets/241a695d-5188-48dc-97fa-063f21217758

**Note:** Translation time in the demo is slightly higher than the benchmark (~4x vs ~3x real time) due to screen recording overhead.

## Requirements

Before setting this up, you **MUST** have Python **3.11**
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

If your CPU and OS support AVX instructions, you can install the deps normally:

                        pip install -r requirements.txt

If you have a device that doesn't support AVX instructions, you'll have to run:

                        pip install -r requirements.txt --no-deps

**NOTE**:  `--no-deps` flags is required for non-AVX CPUs because without it, it'll install jaxlib
which doesn't support non-AVX CPUs and you'll get the error
`RuntimeError: This version of jaxlib was built using AVX instructions, which your CPU and/or operating system`.

If you're a Windows user, you'll have to re install tensorflow so tensorflow-intel gets installed:

                        pip install tensorflow==2.13.1

That package it's not listed because is Windows specific.

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
2. [Model training.](docs/MODEL_TRAINING.md)
3. [Translation.](docs/TRANSLATION.md)
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

I had the chance to run the video translation performance test on different hardware,
this test checks the time it takes to translates a 6 seconds long video at 30 FPS.

These were the results:

### Test on Windows 10 with Intel(R) Celeron(R) CPU G1610 @ 2.60GHz

- Import time = around 5 seconds.
- Video translation time = around 6.3 times slower than the video.
around 4 times slower than the video if it isn't displayed and uses "hands" mode.
(this is not viable)

### Test on Windows 10 with Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz

- Import time = around 5 seconds.
- Video translation time = around 3 times slower than the video.
