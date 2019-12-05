"""Nearest neighbours method.

Typical usage example:

    x_val, languages_characters_indexes_val = load_images(val_folder)

    accuracy = test_nn_accuracy(20, 50, x_val, languages_characters_indexes_val)
"""


import numpy as np
from model_training import make_oneshot_task


def nearest_neighbour_correct(pairs, targets):
    """Checking correction of NN method.

    Args:
        pairs (list<ndarray, ndarray>): Two ndarrays of images, where 1st half belongs
                                        to different categories, 2nd half --- to same.
        targets (ndarray): List of zeros and ones where 1 in place,
                           where photo in 1st list is in the same
                           category as photo in 2nd list.

    Returns:
        1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets).
    """

    L2_distances = np.zeros_like(targets)

    for i in range(len(targets)):
        L2_distances[i] = np.sum(np.sqrt((pairs[0][i] - pairs[1][i])**2))

    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0


def test_nn_accuracy(N, k, x, categories):
    """Returns accuracy of NN approach.

    Args:
        N (int): Number of images in support set.
        k (int): Number of experiments.
        x (ndarray): Array of arrays where each array (character) contains
                     vector images (character representation).
        categories (dict<str, list>): Key is language, value is list
                                      of 2 elements: index of 1st character
                                      and index of last.
    """

    print("Evaluating nearest neighbour on {} unique {}-way one-shot learning tasks ...".format(k, N))

    n_right = 0

    for i in range(k):
        pairs, targets = make_oneshot_task(N, x, categories)
        correct = nearest_neighbour_correct(pairs, targets)
        n_right += correct

    return n_right / k
