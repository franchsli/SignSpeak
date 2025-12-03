"""File to test the perfomance of the SignLanguageTranslator"""

from time import time
from os import path, listdir
from numpy import array


def test_import_performance():
    start = time()
    from translator import SignLanguageTranslator

    end = time()
    print(f"Importing SignLanguageTranslator took {end - start} seconds")

def test_video_translation_performance():
    from translator import SignLanguageTranslator
    # Set the path to the data directory
    PATH = path.abspath("data")
    # Create an array of sign labels by listing the contents of the data directory
    signs = array(listdir(PATH))
    translator = SignLanguageTranslator()
    # load the models down here
    translator.load_model(signs, "words")
    start = time()
    translation = translator.translate_video(path.abspath("data/ADIOS/ADIOS1.mp4"), True)
    end = time()
    print(f"translation: {translation}")
    print(f"Proccessing 6 seconds long video took {end - start} seconds")

def test_image_translation_performance():
    from translator import SignLanguageTranslator
    # Set the path to the data directory
    PATH = path.abspath("letters_data")
    # Create an array of sign labels by listing the contents of the data directory
    signs = array(listdir(PATH))
    translator = SignLanguageTranslator()
    # load the models down here
    translator.load_model(signs, "letters", "models/letterss.keras")
    start = time()
    for _ in range(30):
        translation = translator.translate_image(path.abspath("letters_data/A/A1.png"))
        print(f"translation: {translation}")
    end = time()
    print(f"Time spent translating A1.png 30 times: {end - start} seconds")

def test_live_translation_performance():
    from translator import SignLanguageTranslator
    # Set the path to the data directory
    PATH = path.abspath("data")
    # Create an array of sign labels by listing the contents of the data directory
    signs = array(listdir(PATH))
    translator = SignLanguageTranslator()
    # load the models down here
    translator.load_model(signs, "words")
    start = time()
    translation = translator.translate_video()
    end = time()
    print(f"translation: {translation}")
    print(f"Live translation took {end - start} seconds")

#test_video_translation_performance()
#test_image_translation_performance()
test_live_translation_performance()