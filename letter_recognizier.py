from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples

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


