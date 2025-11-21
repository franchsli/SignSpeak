"""File to test the perfomance of the SignLanguageTranslator"""

from time import time
from os import path, listdir
from numpy import array


def test_import_performance():
    start = time()
    from translator import SignLanguageTranslator

    end = time()
    print(f"Importing SignLanguageTranslator took {end - start} seconds")


