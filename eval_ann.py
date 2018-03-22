import matplotlib.pyplot as plt
import pickle
import sys
import os

DIRNAME = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = os.path.join(DIRNAME, 'out')
MODEL = os.path.join(OUT_DIR, 'ann.hdf5')
HIST = os.path.join(OUT_DIR, 'ann.hist')

if not os.path.exists(MODEL):
    print("missing model:", MODEL)
    sys.exit(1)

if not os.path.exists(HIST):
    print("missing history:", HIST)
    sys.exit(1)


def plot(history):
    # Summarize history for accuracy
    plt.figure()
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Summarize history for loss
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()


if __name__ == '__main__':
    history = pickle.load(open(HIST, "rb"))

    print('val_loss:', min(history['val_loss']))
    print('val_acc:', max(history['val_acc']))

    try:
        plot(history)
    except KeyboardInterrupt as e:
        sys.exit(0)
