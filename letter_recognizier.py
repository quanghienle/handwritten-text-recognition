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

    def create_model(self, load_weights):
        ''' contruct the CNN model
        '''
        model = Sequential()
        model.add(Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu'))
        model.add(Conv2D(32,(3,3),activation='relu'))
        model.add(MaxPool2D())
        model.add(Conv2D(32,(3,3),activation='relu'))
        model.add(Conv2D(16,(3,3),activation='relu'))
        model.add(Flatten())
        model.add(Dense(128,activation='relu'))
        model.add(Dense(27,activation='softmax'))

        model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

        #  print(model.summary())

        if load_weights:
            # path to checkpoint
            checkpoint_path = './model_checkpoints/model_ckpt'
            # load weights
            model.load_weights(checkpoint_path)

        self.model = model

    def fit(self, X_train, Y_train, checkpoint_path='', batch_size=128, epochs=20):
        ''' train the model
        '''
        if checkpoint_path != '':
            # save checkpoint after every epochs
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                verbose=1
            )

            self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=[model_checkpoint_callback])
        else:
            self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

    def evaluate(self, X_test, Y_test):
        ''' Evaluate the model with test data
        '''
        loss, acc = self.model.evaluate(X_test, Y_test)

