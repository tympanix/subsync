import matplotlib.pyplot as plt
import librosa
from train_data import *
import os

TEST_DIR = os.path.join(DIRNAME, 'test')

def spectral_centroid(file):
    y, sr = librosa.load(file)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    plt.figure()
    plt.semilogy(cent.T, label='Spectral centroid')
    plt.ylabel('Hz')
    plt.xticks([])
    plt.xlim([0, cent.shape[-1]])
    plt.legend()
    plt.title('log Power spectrogram')
    plt.tight_layout()


def plot_pred(pred, actual):
    plt.figure()
    plt.plot(pred)
    plt.plot(actual)
    plt.title('prediction evaluation')
    plt.ylabel('label')
    plt.xlabel('time')
    plt.legend(['pred', 'actual'], loc='upper left')


if __name__ == '__main__':
    filename = 'test_440hz_880hz'
    audio = os.path.join(TEST_DIR, filename + '.wav')
    sub = os.path.join(TEST_DIR, filename + '.srt')

    files = [(audio, sub)]
    mfcc, srt = extract_features(files=files)


    for X, Y in zip(mfcc, srt):
        shape = (len(X), 1)
        print("Len", len(X[0]))
        plot_pred(Y, np.array([]))
        spectral_centroid(audio)

        plt.show()