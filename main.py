"""Code to know which is the smallest number of frames per dataset label"""
import os

smallest = 10_000_000
biggest = 0
for folder in os.listdir(os.path.abspath("test")):
    counter = 0
    for file in os.listdir(os.path.join(os.path.abspath("test"), folder)):
        counter += 1
    if counter > biggest:
        biggest = counter
    elif counter < smallest:
        smallest = counter

print(f"The smallest folder has {smallest} frames.\nThe biggest folder has {biggest} frames.") 
    