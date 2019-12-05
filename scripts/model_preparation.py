"""Preparing and initializing model.

Typical usage example:

    model = get_siamese_model((105, 105, 1))
"""


import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K


def initialize_weights(shape, dtype=None):
    """
    The paper, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01.
    """
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape, dtype=None):
    """
    The paper, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01.
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def get_siamese_model(input_shape):
    """
    Model architecture based on the one provided in:
    https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf.
    """

    # Define the tensors for the two input images.
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network.
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    # Generate the encodings (feature vectors) for the two images.
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings.
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score.
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs.
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # Choosing optimizer.
    optimizer = Adam(lr=0.00006)

    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)

    return siamese_net
