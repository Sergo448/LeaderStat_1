# LeaderStat_1/init/Third_MakeModel.py

from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.constraints import maxnorm
import tensorflow as tf
import numpy as np
import idx2numpy

emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103,
                 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]


def emnist_model():
    """

    :return:

    Нейронная сеть соответственно, имеет 62 выхода, на входе она будет получать изображения 28х28,
    после распознавания «1» будет на соответствующем выходе сети.

    Создаем модель сети.
    """
    model = Sequential()
    model.add(
        Convolution2D(filters=32, kernel_size=(3, 3), padding='valid',
                      input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


"""
Как можно видеть, это классическая сверточная сеть,
выделяющая определенные признаки изображения (количество фильтров 32 и 64),
к «выходу» которой подсоединена «линейная» сеть MLP, формирующая окончательный результат.

Обучение нейронной сети

Переходим к самому продолжительному этапу — обучению сети.
Для этого мы возьмем базу EMNIST, скачать которую можно по ссылке (размер архива 536Мб).

Для чтения базы воспользуемся библиотекой idx2numpy. Подготовим данные для обучения и валидации
"""


emnist_path = '/home/sergey/PycharmProjects/LeaderStat_1/init/emnists'
X_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-labels-idx1-ubyte')

X_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-images-idx3-ubyte')
y_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-labels-idx1-ubyte')

X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(emnist_labels))

k = 10
X_train = X_train[:X_train.shape[0] // k]
y_train = y_train[:y_train.shape[0] // k]
X_test = X_test[:X_test.shape[0] // k]
y_test = y_test[:y_test.shape[0] // k]

# Normalize
X_train = X_train.astype(np.float32)
X_train /= 255.0
X_test = X_test.astype(np.float32)
X_test /= 255.0

x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))

# Set a learning rate reduction
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                            patience=3,
                                                            verbose=1,
                                                            factor=0.5,
                                                            min_lr=0.00001)

model = emnist_model().fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat),
                           callbacks=[learning_rate_reduction],
                           batch_size=64, epochs=30)

model.save('emnist_letters.h5')
