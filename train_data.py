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

TRAIN_DIR = 'training'

FREQ = 16000        # Audio frequency
N_MFCC = 13
HOP_LEN = 1024.0     # Num of items per sample
                    # 1 item = 1/16000 seg = 32 ms
ITEM_TIME = (1.0/FREQ)*HOP_LEN


LEN_SAMPLE = 0.5                        # Length in seconds for the input samples
STEP_SAMPLE = 0.25                      # Space between the begining of each sample
LEN_MFCC = LEN_SAMPLE/(HOP_LEN/FREQ)    #  Num of samples to get LEN_SAMPLE
STEP_MFCC = STEP_SAMPLE/(HOP_LEN/FREQ)  #  Num of samples to get STEP_SAMPLE


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
def timeToPos(t, step_mfcc=STEP_MFCC, freq=FREQ, hop_len=HOP_LEN):
    return int((float(freq*timeToSec(t))/hop_len)/step_mfcc)

# Return seconds from cell position
def secToPos(t, step_mfcc=STEP_MFCC, freq=FREQ, hop_len=HOP_LEN):
    return int((float(freq*t)/hop_len)/step_mfcc)

# Return cell position from timestamp
def posToTime(pos, step_mfcc=STEP_MFCC, freq=FREQ, hop_len=HOP_LEN):
    return float(pos)*step_mfcc*hop_len/freq


"""
Uses ffmpeg to transcode and extract audio from movie files in the training
directory. Function returns a list of tuples; the .wav files and corresponding
.srt files to processing
"""
def transcode_audio():
    files = os.listdir(TRAIN_DIR)
    p = re.compile('.*\.[mkv|avi]')
    files = [ f for f in files if p.match(f) ]

    training = []

    for f in files:
        name, extension = os.path.splitext(f)
        input = os.path.join(TRAIN_DIR, f)
        output = os.path.join(TRAIN_DIR, name + '.wav')
        srt = os.path.join(TRAIN_DIR, name + '.srt')

        if not os.path.exists(srt):
            print("missing subtitle for training:", srt)
            sys.exit(1)

        training.append((output, srt))

        if os.path.exists(output):
            continue

        print("Transcoding:", input)
        command = "ffmpeg -y -i {0} -ab 160k -ac 2 -ar {2} -vn {1}".format(input, output, FREQ)
        subprocess.call(command, stderr=subprocess.DEVNULL, shell=True)

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
        print("Processing MFCC:", wav)
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
