from keras.datasets import mnist
from autokeras import ImageClassifier
from autokeras.constant import Constant
import numpy as np

def flip_labels_C(corruption_prob):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(1)

    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_labels = len(y_train)

    indices = np.arange(len(y_train))
    np.random.shuffle(indices)

    x_train = x_train[indices]
    y_train = y_train[indices].astype(np.long)

    corruptive_flags = np.zeros(num_labels)

    gold_fraction = 0.05
    num_classes = 10
    num_gold = int(num_labels * gold_fraction)
    num_silver = num_labels - num_gold
    corruption_matrix = flip_labels_C(0)

    for i in range(num_silver):
        y_train[i] = np.random.choice(num_classes, p=corruption_matrix[y_train[i]])
        corruptive_flags[i] = 1

    y_train = np.concatenate((y_train.reshape((-1, 1)), corruptive_flags.reshape((-1, 1))), axis=1)

    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=30 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)

    print(y * 100)
