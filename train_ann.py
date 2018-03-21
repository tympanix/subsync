from train_data import *
from keras.utils import plot_model
import numpy as np

from keras.layers import Dense, Input, LSTM, Conv1D, Conv2D, Dropout, Flatten, Activation, MaxPooling2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, RMSprop

import matplotlib.pylab as plt


# Conv-1D architecture. Just one sample as input
def ann_model(input_shape):

    inp = Input(shape=input_shape)
    model = inp

    model = Conv1D(filters=12, kernel_size=(3), activation='relu')(model)
    model = Conv1D(filters=12, kernel_size=(3), activation='relu')(model)
    model = Flatten()(model)

    model = Dense(56)(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.2)(model)
    model = Dense(28)(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)

    model = Dense(1)(model)
    model = Activation('sigmoid')(model)

    model = Model(inp, model)
    return model

def train_ann():
    X, Y = extract_features()

    model = ann_model(len(X))

    checkpoint = ModelCheckpoint(filepath=filename, monitor='val_loss', verbose=0, save_best_only=True)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    plot_model(model, to_file='model.png')
    hist = model.fit(X, Y, epochs=2000, batch_size=32, shuffle=True, validation_split=0.3, verbose=0, callbacks=[checkpoint])

    print('val_loss:', min(hist.history['val_loss']))
    print('val_acc:', max(hist.history['val_acc']))


if __name__ == '__main__':
    train_ann()
    print("module used to train the artifical neural network")