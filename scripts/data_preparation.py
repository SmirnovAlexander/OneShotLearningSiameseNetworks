"""Prepare images to processing.

Typical usage example:

    x_val, y_val, languages_characters_indexes_val = load_images(val_folder)

    save_data("val", x_val, languages_characters_indexes_val)

    x, categories = load_data("train")
"""


import os
import pickle
import numpy as np
from imageio import imread


def load_images(path, n=0):
    """Loading alphabets characters images into arrays.

    Returns:
        x (ndarray): Array of arrays where each array (character) contains
                     vector images (character representation).
        languages_characters_indexes (dict<str, list>): Key is language, value is list
                                                        of 2 elements: index of 1st character
                                                        and index of last.
    """

    x = []
    y = []
    languages_characters_indexes = {}
    counter = n

    # Reading all alphabets.
    for alphabet in os.listdir(path):

        print("Loading alphabet: " + alphabet)

        languages_characters_indexes[alphabet] = [counter, None]
        alphabet_path = os.path.join(path, alphabet)

        # Reading all letters.
        for letter in os.listdir(alphabet_path):

            category_images = []
            letter_path = os.path.join(alphabet_path, letter)

            # Reading all letters samples.
            for filename in os.listdir(letter_path):

                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
                category_images.append(np.stack(image))
                y.append(counter)

            try:
                x.append(np.stack(category_images))
            except ValueError as e:
                print(e)
                print("Error in :", category_images)

            counter += 1
            languages_characters_indexes[alphabet][1] = counter - 1

    y = np.vstack(y)
    x = np.stack(x)
    return x, languages_characters_indexes


def save_data(filename, x, languages_characters_indexes, path="../data/arrays"):
    with open(os.path.join(path, filename + ".pickle"), "wb") as f:
        pickle.dump((x, languages_characters_indexes), f)


def load_data(filename, path="../data/arrays", verbose=1):
    with open(os.path.join(path, filename + ".pickle"), "rb") as f:
        (x_train, train_classes) = pickle.load(f)

    if verbose == 1:
        print("Loaded " + filename + " alphabets: \n")
        print(list(train_classes.keys()))
        print("\n")

    return x_train, train_classes
