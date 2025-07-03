"""File to test the perfomance of the SignLanguageTranslator"""
from time import time
from os import path, listdir
from numpy import array

def test_import_performance():
    start = time()
    from translator import SignLanguageTranslator
    end = time()
    print(f"Importing SignLanguageTranslator took {end - start} seconds")

def test_normal_translation_performance():
    from translator import SignLanguageTranslator
    start = time()
    # Set the path to the data directory
    PATH = path.abspath('data')
    # Create an array of action labels by listing the contents of the data directory
    actions = array(listdir(PATH))
    translator = SignLanguageTranslator(actions)
    x = translator.translate_video(path.abspath("data/ADIOS/ADIOS1.mp4"), False)
    print(f"Translation {x}")
    end = time()
    print(f"Proccessing 6 seconds long video took {end - start} seconds")

test_import_performance()