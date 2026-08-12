"""Functions to test the different translations' performances"""

from os import listdir
from time import time


def test_import_performance():
    start = time()
    from core.translator import SignLanguageTranslator  # noqa: F401

    end = time()
    print(f"Importing SignLanguageTranslator took {end - start} seconds")


def test_video_translation_performance():
    from core.translator import SignLanguageTranslator

    with SignLanguageTranslator() as translator:
        # Create an array of sign labels by listing the contents of the data directory
        signs = listdir("data")
        # load the models down here
        translator.load_model(signs, "words", "models/CSL_words_model.keras")
        start = time()
        translation = translator.translate_video("data/ADIOS/ADIOS1.mp4")
        end = time()
        print(f"translation: {translation}")
        print(f"Proccessing 6 seconds long video took {end - start} seconds")


def test_image_translation_performance():
    from core.translator import SignLanguageTranslator

    with SignLanguageTranslator() as translator:
        # Create an array of sign labels by listing the contents of the data directory
        signs = listdir("letters_data")
        # load the models down here
        translator.load_model(signs, "letters", "models/CSL_letters_model.keras")
        start = time()
        for _ in range(30):
            translation = translator.translate_image("letters_data/A/A1.png")
            print(f"translation: {translation}")
        end = time()
        print(f"Time spent translating A1.png 30 times: {end - start} seconds")


def test_live_translation_performance():
    from core.translator import SignLanguageTranslator

    with SignLanguageTranslator() as translator:
        # Create an array of sign labels by listing the contents of the data directory
        signs = listdir("data")
        # load the models down here
        translator.load_model(signs, "words", "models/CSL_words_model.keras")
        start = time()
        translation = translator.translate_video()
        end = time()
        print(f"translation: {translation}")
        print(f"Live translation took {end - start} seconds")
