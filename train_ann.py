from train_data import *
import numpy as np
import pickle
import os

# Keras imports
from keras.layers import Dense, Input, LSTM, Conv1D, Conv2D, Dropout, Flatten, Activation, MaxPooling2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, RMSprop

# Matplotlib imports
import matplotlib.pylab as plt

# Sklearn imports
from sklearn.utils import class_weight

OUT_DIR = "./out"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

"""
Returns a neural network model which can be used for training. The model first
needs to be compiled
"""
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


"""
Trains the neural network using generated test data. Saved the model and the
training history in the ./out folder
"""
def train_ann():
    X, Y = extract_features()

    # Only consider first media file for now
    X, Y = X[0], Y[0]

    shape = (len(X), 1)
    model = ann_model(shape)

    filename = "out/ann.hdf5"

    checkpoint = ModelCheckpoint(filepath=filename, monitor='val_loss', verbose=0, save_best_only=True)
    cutoff = EarlyStopping(monitor='val_loss', min_delta=0.00001, verbose=0, mode='min', patience=5)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    X = X.T
    X = X[..., np.newaxis]

    cw = class_weight.compute_class_weight('balanced', np.unique(Y), Y)

    options = {
        'epochs': 2000,
        'batch_size': 32,
        'class_weight': cw,
        'shuffle': True,
        'validation_split': 0.3,
        'verbose': 0,
        'callbacks': [checkpoint, cutoff]
    }

    hist = model.fit(X, Y, **options)

    print('val_loss:', min(hist.history['val_loss']))
    print('val_acc:', max(hist.history['val_acc']))


    with open('out/ann.hist', 'wb') as hist_file:
        pickle.dump(hist.history, hist_file)


if __name__ == '__main__':
    train_ann()
    print("module used to train the artifical neural network")
