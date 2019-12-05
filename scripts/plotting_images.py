"""Plotting image and images it's compared to.

Typical usage example:

    x, languages_characters_indexes = load_images(train_folder)
    pairs, targets = make_oneshot_task(16, x, languages_characters_indexes, "Sanskrit")

    plot_oneshot_task(pairs)
"""


import numpy as np
import matplotlib.pyplot as plt


def concat_images(X):
    """Concatenates a bunch of images into a big matrix for plotting purposes.
    """
    nc, h, w, _ = X.shape
    X = X.reshape(nc, h, w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n * w, n * h))
    x = 0
    y = 0
    for example in range(nc):
        img[x * w:(x + 1) * w, y * h:(y + 1) * h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img


def plot_oneshot_task(pairs):
    """Plotting image and images it's compared to.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.matshow(pairs[0][0].reshape(105, 105), cmap='gray')
    img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
