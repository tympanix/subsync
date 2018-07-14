#!/usr/bin/env python
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import subprocess
import librosa
import pysrt
import sys
import os
import re

DIRNAME = os.path.dirname(os.path.realpath(__file__))
TRAIN_DIR = os.path.join(DIRNAME, 'training')

FREQ = 16000        # Audio frequency
N_MFCC = 13
HOP_LEN = 512.0    # Num of items per sample
                    # 1 item = 1/16000 seg = 32 ms
ITEM_TIME = HOP_LEN/FREQ


if not os.path.exists(TRAIN_DIR):
    print("missing training data in directory:", TRAIN_DIR)
    sys.exit(1)

# Convert timestamp to seconds
def timeToSec(t):
    total_sec = float(t.milliseconds)/1000
    total_sec += t.seconds
    total_sec += t.minutes*60
    total_sec += t.hours*60*60
    return total_sec

# Return timestamp from cell position
def timeToPos(t, freq=FREQ, hop_len=HOP_LEN):
    return round(timeToSec(t)/(hop_len/freq))


"""
Uses ffmpeg to transcode and extract audio from movie files in the training
directory. Function returns a list of tuples; the .wav files and corresponding
.srt files to processing
"""
def transcode_audio(dir=TRAIN_DIR):
    files = os.listdir(dir)
    p = re.compile('.*\.[mkv|avi|mp4]')
    files = [ f for f in files if p.match(f) ]

    training = []

    for f in files:
        name, extension = os.path.splitext(f)
        input = os.path.join(dir, f)
        output = os.path.join(dir, name + '.wav')
        srt = os.path.join(dir, name + '.srt')

        if not os.path.exists(srt):
            print("missing subtitle for training:", srt)
            sys.exit(1)

        training.append((output, srt))

        if os.path.exists(output):
            continue

        print("Transcoding:", input)
        command = "ffmpeg -y -i {0} -ab 160k -ac 2 -ar {2} -vn {1}".format(input, output, FREQ)
        code = subprocess.call(command, stderr=subprocess.DEVNULL, shell=True)
        if code != 0:
            raise Exception("ffmpeg returned: {}".format(code))

    return training


"""
Extracts the features and labels from the .wav and .srt file. The audio is
processed using MFCC. Returns a tuple where the first element is the MFCC data
and the second argument is the labels for the data.
"""
def extract_features(files=None):
    if files is None:
        files = transcode_audio()

    audio = []
    labels = []

    for (wav, srt) in files:
        print("Processing audio:", wav)
        y, sr = librosa.load(wav, sr=FREQ)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(HOP_LEN), n_mfcc=int(N_MFCC))
        label = extract_labels(srt, len(mfcc[0]))
        audio.append(mfcc)
        labels.append(label)

    return audio, labels


"""
Processes a .srt file and returns a numpy array of labels for each sample. If
there is a subtitle at the i'th sample, there is a 1 at position i, else 0.
"""
def extract_labels(srt, samples):
    subs = pysrt.open(srt)
    labels = np.zeros(samples)
    for sub in subs:
        start = timeToPos(sub.start)
        end = timeToPos(sub.end)+1
        for i in range(start, end):
            if i < len(labels):
                labels[i] = 1

    return labels


"""
Returns a mask of indexes in Y (a selection) for which the selection has an
equal/balanced choice for every unique value in Y. That is exactly n items are
chosen for each class.
"""
def balance_classes(Y):
    uniq = np.unique(Y)
    C = [np.squeeze(np.argwhere(Y==c)) for c in uniq]
    minority = min([len(c) for c in C])
    M = [np.random.choice(c, size=minority, replace=False) for c in C]
    return np.append(*M)


"""
Prepares the data for processing in a neural network. First the data is
converted to the proper dimensions, and afterwards the data is balanced
for class imbalance issues.
"""
def prepare_data(X, Y, balance=True):
    X = X.T
    X = X[..., np.newaxis]

    if balance:
        # Balance classes such that there are n of each class
        balance = balance_classes(Y)
        X = X[balance]
        Y = Y[balance]

    return X, Y


"""
Used to plot the MFCC spectrograms for inspecting
"""
def plot_mfcc(mfcc):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mfccs, labels = extract_features()

    for mfcc in mfccs:
        plot_mfcc(mfcc)
