import sklearn
import numpy as np
from train_data import *
from train_ann import *

MODEL = os.path.join(OUT_DIR, 'ann.hdf5')

if not os.path.exists(MODEL):
    print("missing model:", MODEL)
    sys.exit(1)


def logloss(pred, actual):
    begin = np.argmax(actual) * (-1)
    end = np.argmax(actual[::-1]) + 1
    print("Calculating {} logloss values".format(end-begin))
    logloss = np.zeros(end-begin)
    indices = np.zeros(end-begin)
    for i, offset in enumerate(range(begin, end)):
        logloss[i] = sklearn.metrics.log_loss(np.roll(actual, offset), pred)
        indices[i] = offset

    return indices, logloss


def plot_logloss(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.title('logloss over shifts')
    plt.ylabel('logloss')
    plt.xlabel('shifts')
    plt.legend(['logloss'], loc='upper left')


def load_model(input_shape):
    model = ann_model(input_shape)
    model.load_weights(MODEL)
    return model

if __name__ == '__main__':
    files = transcode_audio()
    mfcc, labels = extract_features(files=files)

    for X, Y in zip(mfcc, labels):
        shape = (len(X), 1)
        X, Y = prepare_data(X, Y, balance=False)
        model = load_model(shape)
        print("Predicting...")
        pred = model.predict(X, batch_size=32)
        print("Done...")
        x, y = logloss(pred, Y)
        plot_logloss(x, y)
        plt.show()
