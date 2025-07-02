"""File to test the perfomance of the SignLanguageTranslator"""
from time import time

def test_import_performance():
    start = time()
    from translator import SignLanguageTranslator
    end = time()
    print(f"Importing SignLanguageTranslator took {end - start} seconds")

test_import_performance()