import os

import cv2
import numpy as np

from core.load import create_data, download_dataset, DIGITS_URL
from core.models import Model
from core.layers import Layer_Dense, Layer_Dropout
from core.accuracy import Accuracy_Regression, Accuracy_Categorical
from core.activations import Activation_ReLU, Activation_Linear, Activation_Softmax
from core.losses import Loss_MeanSquaredError, Loss_CategoricalCrossentropy, \
    Activation_Softmax_Loss_CategoricalCrossentropy
from core.optimizers import Optimizer_Adam
from core.predictor import App


def test_fashion():
    # Create dataset
    X, y, X_test, y_test = create_data('/home/vroman11/my_scripts/LUST/data/mnist_fashion/train',
                                       '/home/vroman11/my_scripts/LUST/data/mnist_fashion/test')

    # Shuffle the training dataset
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    # Scale and reshape samples
    X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    # Instantiate the model
    model = Model()
    # Add layers
    model.add(Layer_Dense(X.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    # Set loss, optimizer and accuracy objects
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(decay=1e-4),
        accuracy=Accuracy_Categorical()
    )

    # Finalize the model
    model.finalize()
    # Train the model
    model.train(X, y, validation_data=(X_test, y_test), epochs=5, batch_size=128, print_every=100)

    model.save('fashion.model')


def test_digits():
    download_dataset(DIGITS_URL, 'digits.zip', 'data')
    # Create dataset
    X, y, X_test, y_test = create_data(
        'data/MNIST Dataset JPG format/MNIST - JPG - training',
        'data/MNIST Dataset JPG format/MNIST - JPG - testing')

    # Shuffle the training dataset
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    # Scale and reshape samples
    X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    # Instantiate the model
    model = Model()
    # Add layers
    model.add(Layer_Dense(X.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    # Set loss, optimizer and accuracy objects
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(decay=1e-4),
        accuracy=Accuracy_Categorical()
    )

    # Finalize the model
    model.finalize()
    # Train the model
    model.train(X, y, validation_data=(X_test, y_test), epochs=5, batch_size=128, print_every=100)

    model.save('digits.model')


if __name__ == '__main__':
    if not os.path.isfile('digits.model'):
        test_digits()
    model = Model.load('digits.model')
    App(model)


