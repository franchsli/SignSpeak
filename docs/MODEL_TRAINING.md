# Model training

After you have your dataset ready, you must train a model using the
respective class. The same class you used to create the dataset is the
same you'll use to train your model calling the method .train().

## Images

``` python3
# main.py in root
from core.data_handlers import ImageHandler
ImageSignHandler = ImageHandler("data")
# datasets creation code
...
ImageSignHandler.train("dataset", "models/letters_model.keras")
```

## Videos

``` python3
# main.py in root
from core.data_handlers import VideoHandler
VideoSignHandler = VideoHandler("data")
# datasets creation code
...
VideoSignHandler.train("dataset", "models/words_model.keras")
```

If you didn't create a proper dataset with any of these classes
but you still have a proper dataset for either [videos](DATASET_CREATION.md#L50-L62) or [images](DATASET_CREATION.md#L84-L96)
you can still use it with the respective class.

**NOTE:** It is important that you use the respective class for this,
because images and videos are treated differently and they are designed for
different purposes, images are for letters and videos for words, concepts, etc.

## Model naming convention

This is the recommended convetion for naming the models you create with .train():

1. Sign language
2. Model's target prediction
3. "model"

Where 1 is "CSL" or "ASL", etc..., 2 is "words" or "letters".
Example: "CSL_words_model.keras"

This is *recommended*, *not* mandatory, you can name it whatever you feel like, but this
convention helps you stay organized.
The class does not care about this, if the path is correct it will store it as you'd like
(see the code snippets above for instance).

After you have a trained model, you can [translate](TRANSLATION.MD).
