from time import time
from os import listdir, path
from numpy import array
from translator import SignLanguageTranslator
from predictor import GesturePredictor
start = time()
# Set the path to the data directory
PATH = path.abspath('data')
# Create an array of action labels by listing the contents of the data directory
actions = array(listdir(PATH))
predictor = GesturePredictor()
# load the models down here
predictor.load_model(actions, "words")
# load the predictor instance in the translator
translator = SignLanguageTranslator(predictor)
translation = translator.translate_video(path.abspath("data/ADIOS/ADIOS1.mp4"), True)
print(translation)
end = time()
print(f"Proccessing 6 seconds long video took {end - start} seconds")