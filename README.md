# SignSpeak

A sign language translator.

## TODO

***IMPORTANT***

- Performance:
  current import time = around 5 seconds.
  current translation time = around 6.3 times slower.

### Low priority

- Break down translate_video() into smaller functions (check Claude Chat [the first one]).
- Fully rework VideoHandler, GestureHandler, etc.
  - These classes will only handle datasets creation and model training, nothing else.
  Each class will specialize in handling one format (video, image, or webcam [think about this])
- Use VideoHandler functionalities into WebCamHandler.
- Implement an ImageHandler (think about it).
- Collect more data (sentences and letters) to train the VideoHandler with.

[use this repo](https://github.com/dgovor/Sign-Language-Translator)

## General Resources

[Main dataset used](https://www.kaggle.com/datasets/juanrrai/10-words-slc-and-3-people)

[Sign language processing](https://pypi.org/project/sign-language-tools/)

[Text to Sign language library docs](https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators)

[Datasets](https://www.kaggle.com/datasets?search=colombian+sign+language)

[More Data](https://www.youtube.com/watch?v=JMraBJsA9oI&list=PLI7rDimYXOdhyty-lEXsxQgiLfYKnnqmY&index=4)

[Very good dataset](https://bivl2ab.uis.edu.co/dataset-info)
