"""File to test the perfomance of the SignLanguageTranslator"""

from time import time
from os import listdir


def test_import_performance():
    start = time()
    from core.translator import SignLanguageTranslator

    end = time()
    print(f"Importing SignLanguageTranslator took {end - start} seconds")


def test_video_translation_performance():
    from core.translator import SignLanguageTranslator

    # Create an array of sign labels by listing the contents of the data directory
    signs = listdir("data")
    translator = SignLanguageTranslator()
    # load the models down here
    translator.load_model(signs, "words")
    start = time()
    translation = translator.translate_video("data/ADIOS/ADIOS1.mp4")
    end = time()
    print(f"translation: {translation}")
    print(f"Proccessing 6 seconds long video took {end - start} seconds")


def test_image_translation_performance():
    from core.translator import SignLanguageTranslator

    # Create an array of sign labels by listing the contents of the data directory
    signs = listdir("letters_data")
    translator = SignLanguageTranslator()
    # load the models down here
    translator.load_model(signs, "letters", "models/letterss.keras")
    start = time()
    for _ in range(30):
        translation = translator.translate_image("letters_data/A/A1.png")
        print(f"translation: {translation}")
    end = time()
    print(f"Time spent translating A1.png 30 times: {end - start} seconds")


def test_live_translation_performance():
    from core.translator import SignLanguageTranslator

    # Create an array of sign labels by listing the contents of the data directory
    signs = listdir("data")
    translator = SignLanguageTranslator()
    # load the models down here
    translator.load_model(signs, "words")
    start = time()
    translation = translator.translate_video()
    end = time()
    print(f"translation: {translation}")
    print(f"Live translation took {end - start} seconds")
