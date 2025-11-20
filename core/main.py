from time import time
from os import listdir, path
from numpy import array
from translator import SignLanguageTranslator
from predictor import SignPredictor

def test_video_translation_performance():
    start = time()
    # Set the path to the data directory
    PATH = path.abspath("data")
    # Create an array of sign labels by listing the contents of the data directory
    signs = array(listdir(PATH))
    predictor = SignPredictor()
    # load the models down here
    predictor.load_model(signs, "words")
    # load the predictor instance in the translator
    translator = SignLanguageTranslator(predictor)
    translation = translator.translate_video(path.abspath("data/ADIOS/ADIOS1.mp4"), True)
    print(f"translation: {translation}")
    end = time()
    print(f"Proccessing 6 seconds long video took {end - start} seconds")

def test_image_translation_performance():
    start = time()
    # Set the path to the data directory
    PATH = path.abspath("letters_data")
    # Create an array of sign labels by listing the contents of the data directory
    signs = array(listdir(PATH))
    predictor = SignPredictor()
    # load the models down here
    predictor.load_model(signs, "letters", "models/letterss.keras")
    # load the predictor instance in the translator
    translator = SignLanguageTranslator(predictor)
    for _ in range(30):
        translation = translator.translate_image(path.abspath("letters_data/A/A1.png"))
        print(f"translation: {translation}")
    end = time()
    print(f"Time spent translating A1.png 30 times: {end - start} seconds")

if __name__ == "__main__":
    test_image_translation_performance()
