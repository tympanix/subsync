import matplotlib.pyplot as plt
from train_ann import *
from train_data import *
import os

MODEL = os.path.join(OUT_DIR, 'ann.hdf5')


if not os.path.exists(MODEL):
    print("missing model:", MODEL)
    sys.exit(1)


def load_model(input_shape):
    model = ann_model(input_shape)
    model.load_weights(MODEL)
    return model


def plot_pred(pred, actual):
    plt.figure()
    plt.plot(pred)
    plt.plot(actual)
    plt.title('prediction evaluation')
    plt.ylabel('label')
    plt.xlabel('time')
    plt.legend(['pred', 'actual'], loc='upper left')


if __name__ == '__main__':
    files = transcode_audio()
    wav, srt = extract_features(files=files)

    for X, Y in zip(wav, srt):
        shape = (len(X), 1)
        X, Y = prepare_data(X, Y, balance=False)
        model = load_model(shape)
        pred = model.predict(X, batch_size=32)
        plot_pred(pred, Y)

    plt.show()
