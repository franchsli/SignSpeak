# Translation

In order to translate, you need to load the models you're
going to use, like this:

``` python3
from translator import SignLanguageTranslator
with SignLanguageTranslator() as translator:
    translator = SignLanguageTranslator()
    translator.load_model(signs, model_name, model_path)
    ...
```

"signs" is a list of strings containing all the signs the model knows
in the same exact order it learned them. If the model was trained
with this dataset:

``` ASCII
.
└── dataset/
    ├── A/
    │   ├── 1.npy
    │   ├── 2.npy
    │   └── ...
    ├── B/
    │   ├── 1.npy
    │   ├── 2.npy
    │   └── ...
    └── ...
```

Then it means the model learned "A" then "B", so when loading the model,
signs should be like this:

``` python3
signs = ["A", "B"]
```

It is important that the signs variable is in the same order the model
learned those signs (which is normally in alphanumeric order), because the model
assigns integers to map the signs (e.g. HELLO = 15) and the model predicts answering with
those integers, if you pass an unordered signs list the translation will be incorrect
since the model predicts an integer but the code tries to look for that index in the given list
no matter what.

Due to that, you should have knoweledge of which signs the model learned,
this won't be a problem if you create your own models.

There are two kinds of translations:

## Image translation

``` python3
# main.py file in root
from os import listdir
from core.translator import SignLanguageTranslator
with SignLanguageTranslator() as translator:
    # Create an array of sign labels by listing the contents of the data directory
    signs = listdir("letters_data")
    # load the models down here
    translator.load_model(signs, "letters", "models/letters.keras")
    translation = translator.translate_image("some_image.png")
    print(f"translation: {translation}")
```

## Video translation

``` python3
# main.py file in root
from os import listdir
from core.translator import SignLanguageTranslator
with SignLanguageTranslator() as translator:
    # Create an array of sign labels by listing the contents of the data directory
    signs = listdir("data")
    # load the models down here
    translator.load_model(signs, "words", "models/words.keras")
    translation = translator.translate_video("some_video.mp4")
    print(f"translation: {translation}")
```

If you want to correct the translations, refer to [text correction](TEXT_CORRECTION.md)
