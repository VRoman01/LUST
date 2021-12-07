# Loads a MNIST dataset
import os
import urllib.request
from zipfile import ZipFile

import cv2
import numpy as np

DIGITS_URL = 'https://github.com/teavanist/MNIST-JPG/raw/master/MNIST%20Dataset%20JPG%20format.zip'
FASHION_URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'


def download_dataset(URL, FILE, FOLDER):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)
    print('Unzipping images...')
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(os.path.join(os.getcwd(), FOLDER))


def load_dataset(path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(path)

    # Create lists for samples and labels
    X = []
    y = []
    #     # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, label)):
            # Read the image
            image = cv2.imread(os.path.join(path, label, file), cv2.IMREAD_UNCHANGED)
            # And append it and a label to the lists
            X.append(image)
            y.append(label)
    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


def create_data(train_path, test_path):
    X, y = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    return X, y, X_test, y_test