"""Training model.

Typical usage example:

    model = get_siamese_model((105, 105, 1))
    x, categories = load_data("train")
    x_val, categories_val = load_data("val")

    pairs, targets = get_batch(8, x, categories)

    pairs, targets = make_oneshot_task(4, x_val, categories_val)

    correct_rate = test_oneshot_task(model, 4, 50, x_val, categories_val)

    train(model, evaluate_every=1)
"""

import os
import time

import numpy as np
import numpy.random as rng
from sklearn.utils import shuffle

import wandb


def get_batch(batch_size, x, categories):
    """Create batch of batch_size pairs, half different class, half same class.

    Args:
        x (ndarray): Array of arrays where each array (character) contains
                     vector images (character representation).
        categories (dict<str, list>): Key is language, value is list
                                      of 2 elements: index of 1st character
                                      and index of last.

    Returns:
        pairs (list<ndarray, ndarray>): Two ndarrays of images, where 1st half belongs
                                        to different categories, 2nd half --- to same.
        targets (ndarray): List of zeros and ones where 1 in place,
                           where photo in 1st list is in the same
                           category as photo in 2nd list.
    """

    n_classes, n_examples, w, h = x.shape

    # Randomly sample several classes to use in the batch.
    categories = rng.choice(n_classes, size=(batch_size,), replace=False)

    # Initialize 2 empty arrays for the input image batch.
    pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

    # Initialize vector for the targets.
    targets = np.zeros((batch_size,))

    # Make one half of it '1's, so 2nd half of batch has same class.
    targets[batch_size // 2:] = 1

    for i in range(batch_size):

        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i, :, :, :] = x[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)

        # Pick images of same class for 1st half, different for 2nd.
        if i >= batch_size // 2:
            category_2 = category
        else:
            # Add a random number to the category module n classes to ensure 2nd image has a different category.
            category_2 = (category + rng.randint(1, n_classes)) % n_classes

        pairs[1][i, :, :, :] = x[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets


def make_oneshot_task(N, x, categories, language=None):
    """Create pairs of test image, support set for testing N-way one-shot learning.

    Args:
        N (int): Number of images in support set.
        x (ndarray): Array of arrays where each array (character) contains
                     vector images (character representation).
        categories (dict<str, list>): Key is language, value is list
                                      of 2 elements: index of 1st character
                                      and index of last.
        language (str): Specify language to pick images
                        only from characters of this
                        language.
    Returns:
        pairs (list<ndarray, ndarray>): First array in pairs list is array of
                                        N photos of 1 chosen test image.
                                        Second array is list of N random photos
                                        where 1 photo's category matching with
                                        1st array photo.
        targets (ndarray): List of zeros and one 1 in place, where
                           photo in 1st pairs list is in the same
                           category as photo in 2nd list.
    """

    n_classes, n_examples, w, h = x.shape

    indices = rng.randint(0, n_examples, size=(N,))

    # If language is specified, select characters for that language.
    if language is not None:
        low, high = categories[language]
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low, high), size=(N,), replace=False)

    # If no language specified just pick a bunch of random letters.
    else:
        categories = rng.choice(range(n_classes), size=(N,), replace=False)

    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))

    test_image = np.asarray([x[true_category, ex1, :, :]] * N).reshape(N, w, h, 1)
    test_image = test_image.astype("float32")

    support_set = x[categories, indices, :, :]
    support_set[0, :, :] = x[true_category, ex2]
    support_set = support_set.reshape(N, w, h, 1)
    support_set = support_set.astype("float32")

    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image, support_set]

    return pairs, targets


def test_oneshot_task(model, N, k, x, categories, verbose=1):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks.

    Args:
        N (int): Number of images in support set.
        k (int): Number of experiments.
        x (ndarray): Array of arrays where each array (character) contains
                     vector images (character representation).
        categories (dict<str, list>): Key is language, value is list
                                      of 2 elements: index of 1st character
                                      and index of last.
    Returns:
        percent_correct (float): Percentage show how many times
                                 maximum similarity ratio corresponds
                                 with actual similarity.
    """

    n_correct = 0

    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))

    for i in range(k):
        inputs, targets = make_oneshot_task(N, x, categories)
        predictions = model.predict(inputs)
        if np.argmax(predictions) == np.argmax(targets):
            n_correct += 1

    correct_rate = n_correct / k

    if verbose:
        print(
            "Got an average of {:.2f} in {} way one-shot learning accuracy \n".format(correct_rate, N))

    return correct_rate


def train(model,
          x,
          categories,
          x_val,
          categories_val,
          model_path="../models/",
          train_name="train",
          val_name="val",
          evaluate_every=100,
          batch_size=32,
          iterations=20000,
          support_set_quantity=20,
          number_of_validations=30):
    """
    Args:
        evaluate_every (int): Interval for evaluating on one-shot tasks.
        iterations (int): No. of training iterations.
        support_set_quantity (int): Number of classes for testing on one-shot tasks.
        n_val (int): Number of one-shot tasks to validate on.
    """

    wandb.init(project="one_shot_learning",
               dir="../models")

    best_val_acc = -1

    print("Starting training process!")
    print("-------------------------------------")

    t_start = time.time()

    for i in range(1, iterations + 1):
        (inputs, targets) = get_batch(batch_size, x, categories)
        loss = model.train_on_batch(inputs, targets)
        if i % evaluate_every == 0:
            print("\n ------------- \n")
            print("Time for {} iterations: {:.2f} mins".format(i, (time.time() - t_start) / 60.0))
            print("Train Loss: {:.2f}".format(loss))

            val_acc = test_oneshot_task(model, support_set_quantity,
                                        number_of_validations, x_val, categories_val, verbose=True)

            model.save_weights(os.path.join(
                model_path, "iteration-{}-val_acc-{:.2f}-loss-{:.2f}.h5".format(i, val_acc, loss)))

            if val_acc >= best_val_acc:
                print("Current best: {:.2f}, previous best: {:.2f}".format(val_acc, best_val_acc))
                best_val_acc = val_acc

            wandb.log({'iteration': i, 'loss': loss, 'val_acc': val_acc})

    return model
