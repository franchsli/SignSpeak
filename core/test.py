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
    from predictor import SignPredictor
    start = time()
    # Set the path to the data directory
    PATH = path.abspath('data')
    # Create an array of sign labels by listing the contents of the data directory
    signs = array(listdir(PATH))
    predictor = SignPredictor()
    # load the models down here
    predictor.load_model(signs, "words")
    # load the predictor instance in the translator
    translator = SignLanguageTranslator(predictor)
    translation = translator.translate_video(path.abspath("data/ADIOS/ADIOS1.mp4"), True)
    print(translation)
    end = time()
    print(f"Proccessing 6 seconds long video took {end - start} seconds")

test_import_performance()