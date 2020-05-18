import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
CIFAR_10_DIRS = 'CIFAR_10'


# official provided function for read data file.
# input:    file(path of data file)
# output:   data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


# Read cifar 10 data set.
# input:    none
# output:   train_data, train_labels, test_data, test_labels (dtype = numpy.array)
# training data set in 5 files: data_batch_1, data_batch_2,..., data_batch_5
# test data set in 1 file:
def read_cifar_10():
    training_set_file_path = [os.path.join(CIFAR_10_DIRS, 'data_batch_' + str(num)) for num in range(1, 6)]
    train_data = np.array(unpickle(training_set_file_path[0])[b'data']).reshape((-1, 3, 32, 32))
    train_data = np.transpose(train_data, (0, 2, 3, 1))
    train_labels = np.array(unpickle(training_set_file_path[0])[b'labels'])
    test_data = np.array(unpickle(os.path.join(CIFAR_10_DIRS, 'test_batch'))[b'data']).reshape((-1, 3, 32, 32))
    test_data = np.transpose(test_data, (0, 2, 3, 1))
    test_labels = np.array(unpickle(os.path.join(CIFAR_10_DIRS, 'test_batch'))[b'labels'])
    for path in training_set_file_path[1:]:
        data_temp = np.array(unpickle(path)[b'data']).reshape((-1, 3, 32, 32))
        data_temp = np.transpose(data_temp, (0, 2, 3, 1))
        labels_temp = np.array(unpickle(path)[b'labels'])
        train_data = np.concatenate([train_data, data_temp], axis=0)
        train_labels = np.concatenate([train_labels, labels_temp], axis=0)
    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
    print(read_cifar_10()[0].shape)
