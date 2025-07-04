from time import time
from os import listdir, path
from numpy import array
from translator import SignLanguageTranslator
start = time()
# Set the path to the data directory
PATH = path.abspath('data')
# Create an array of action labels by listing the contents of the data directory
actions = array(listdir(PATH))
translator = SignLanguageTranslator(actions)
translator.translate_video(path.abspath("data/ADIOS/ADIOS1.mp4"))
end = time()
print(f"Proccessing 6 seconds long video took {end - start} seconds")