from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPool2D,Flatten,Dropout
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt


# this class is uesd to generate the handwritten letters dataset
# in order to train the model
class DataLoader():
    def __init__(self):

        # Load train and test from the EMNIST dataset
        X_train, Y_train = extract_training_samples('letters')
        X_test, Y_test = extract_test_samples('letters')

        # 26 English alphabets in correct order
        self.alphabets = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # reshape the images for the model
        self.X_train = X_train.reshape(-1,28,28,1)
        self.X_test = X_test.reshape(-1,28,28,1)

        self.Y_train = to_categorical(Y_train,num_classes=27)
        self.Y_test = to_categorical(Y_test,num_classes=27)


# this class creates a simple 8-layered Convolutional Neiral Network model
# to classify 26 different handwritten letters
class LetterRecognizer():

    def __init__(self, load_weights=True):
        ''' initialize the model
            if load_weights is true, then the model will use the weights that were previously trained (accuracy of 92%)
        '''
        self.create_model(load_weights)

