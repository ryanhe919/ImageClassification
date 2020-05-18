import tensorflow as tf
from tensorflow.keras import layers, models
from ImageClassification import data_process
import os


class CNN:
    def __init__(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2), strides=2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])
        self.model.summary()

    def fit(self, train_data, train_labels, test_data, test_labels):
        self.model.compile(optimizer='adam',
                           loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.fit(train_data, train_labels, epochs=10,
                       validation_data=(test_data, test_labels))


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = data_process.read_cifar_10()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1, 0'
    cnn = CNN()
    cnn.fit(train_data, train_labels, test_data, test_labels)
