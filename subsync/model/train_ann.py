from train_data import *
import numpy as np
import pickle
import os

# Keras imports
from keras.layers import (
    Dense,
    Input,
    LSTM,
    Conv1D,
    Conv2D,
    Dropout,
    Flatten,
    Activation,
    MaxPooling2D,
)
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, RMSprop

# Matplotlib imports
import matplotlib.pylab as plt

# Sklearn imports
from sklearn.utils import class_weight

DIRNAME = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = os.path.join(DIRNAME, "out")

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

"""
Returns a neural network model which can be used for training. The model first
needs to be compiled
"""


def ann_model(input_shape):
    inp = Input(shape=input_shape, name="mfcc_in")
    model = inp

    model = Conv1D(filters=12, kernel_size=(3), activation="relu")(model)
    model = Conv1D(filters=12, kernel_size=(3), activation="relu")(model)
    model = Flatten()(model)

    model = Dense(56)(model)
    model = Activation("relu")(model)
    model = BatchNormalization()(model)
    model = Dropout(0.2)(model)
    model = Dense(28)(model)
    model = Activation("relu")(model)
    model = BatchNormalization()(model)

    model = Dense(1)(model)
    model = Activation("sigmoid")(model)

    model = Model(inp, model)
    return model


"""
Trains the neural network using generated test data. Saved the model and the
training history in the ./out folder.
"""


def train_ann():
    X, Y = extract_features()

    # Only consider first media file for now
    X, Y = X[0], Y[0]

    shape = (len(X), 1)
    model = ann_model(shape)

    filename = "out/ann.hdf5"

    checkpoint = ModelCheckpoint(
        filepath=filename, monitor="val_loss", verbose=0, save_best_only=True
    )
    cutoff = EarlyStopping(monitor="val_loss", min_delta=1e-3, mode="min", patience=5)

    model.compile(
        loss="mean_squared_error", optimizer=Adam(lr=0.001), metrics=["accuracy"]
    )

    X, Y = prepare_data(X, Y, balance=True)

    print("Label 1:", len(Y[Y == 1]))
    print("Label 0:", len(Y[Y == 0]))

    # Permutate training data in random order
    rand = np.random.permutation(np.arange(len(Y)))
    X = X[rand]
    Y = Y[rand]

    options = {
        "epochs": 200,
        "batch_size": 32,
        "shuffle": True,
        "validation_split": 0.3,
        "verbose": 2,
        "callbacks": [checkpoint, cutoff],
    }

    print("Training neural network:", filename)
    hist = model.fit(X, Y, **options)

    print("val_loss:", min(hist.history["val_loss"]))
    print("val_acc:", max(hist.history["val_acc"]))

    with open("out/ann.hist", "wb") as hist_file:
        pickle.dump(hist.history, hist_file)


if __name__ == "__main__":
    train_ann()
    print("module used to train the artifical neural network")
