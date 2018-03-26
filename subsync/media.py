import os
import librosa
import subprocess
import tempfile
import soundfile as sf
import io
import pysrt

import matplotlib.pyplot as plt
import numpy as np
import sklearn


class Media:
    """
    Media class represents a media file on disk for which the content can be
    analyzed and retrieved.
    """

    # List of supported media formats
    FORMATS = ['.mkv', '.mp4', '.wmv', '.avi', '.flv']

    # The frequency of the generated audio
    FREQ = 16000

    # The command used to generate raw wav audio with ffmpeg
    FFMPEG_CMD = 'ffmpeg -y -i "{0}" -ab 160k -ac 2 -ar {2} -vn "{1}"'

    # The number of coefficients to extract from the mfcc
    N_MFCC = 13

    # The number of samples in each mfcc coefficient
    HOP_LEN = 512.0

    # The length (seconds) of each item in the mfcc analysis
    LEN_MFCC = HOP_LEN/FREQ


    def __init__(self, filepath):
        prefix, ext = os.path.splitext(filepath)
        if ext not in Media.FORMATS:
            raise ValueError('filetype {} not supported'.format(ext))
        self.filepath = filepath
        self.filename = os.path.basename(prefix)
        self.extension = ext


    def subtitles(self):
        dir = os.path.dirname(self.filepath)
        for f in os.listdir(dir):
            if f.endswith('.srt') and f.startswith(self.filename):
                yield Subtitle(self, os.path.join(dir, f))


    def mfcc(self):
        outpath = None
        with tempfile.NamedTemporaryFile(suffix='.wav', mode='r') as output:
            outpath = output.name
        print("Transcoding:", output.name)
        command = Media.FFMPEG_CMD.format(self.filepath, outpath, Media.FREQ)
        code = subprocess.call(command, stderr=subprocess.DEVNULL, shell=True)
        y, sr = librosa.load(outpath, sr=Media.FREQ)
        print("Analysing...")
        self.mfcc = librosa.feature.mfcc(y=y, sr=sr,
            hop_length=int(Media.HOP_LEN),
            n_mfcc=int(Media.N_MFCC)
        )
        os.remove(outpath)
        return self.mfcc



class Subtitle:
    """
    Subtitle class represnets an .srt file on disk and provides
    functionality to inspect and manipulate the subtitle content
    """

    # The maximum number of seconds to shift
    MAX_TIME = 12.0

    # Maximum number of items to shift
    MAX_SHIFTS = int(MAX_TIME/(Media.HOP_LEN/Media.FREQ))


    def __init__(self, media, path):
        self.media = media
        self.path = path

    def labels(self):
        if self.media.mfcc is None:
            raise RuntimeError("Must analyse mfcc before generating labels")
        subs = pysrt.open(self.path)
        samples = len(self.media.mfcc[0])
        labels = np.zeros(samples)
        for sub in subs:
            start = timeToPos(sub.start)
            end = timeToPos(sub.end)+1
            for i in range(start, end):
                if i < len(labels):
                    labels[i] = 1

        return labels


    def logloss(self, pred, actual):
        begin = np.argmax(actual) * (-1)
        begin = max(begin, Subtitle.MAX_SHIFTS*(-1))
        end = np.argmax(actual[::-1]) + 1
        end = min(end, Subtitle.MAX_SHIFTS)
        print("Calculating {} logloss values...".format(end-begin))
        logloss = np.zeros(end-begin)
        indices = np.zeros(end-begin)
        for i, offset in enumerate(range(begin, end)):
            logloss[i] = sklearn.metrics.log_loss(np.roll(actual, offset), pred)
            indices[i] = offset

        return indices, logloss


    def sync(self, net):
        labels = self.labels()
        mfcc = self.media.mfcc.T
        mfcc = mfcc[..., np.newaxis]
        pred = net.predict(mfcc)
        x, y = self.logloss(pred, labels)
        self.plot_logloss(x, y)


    def plot_logloss(self, x, y):
        plt.figure()
        plt.plot(x, y)
        plt.title('logloss over shifts')
        plt.ylabel('logloss')
        plt.xlabel('shifts')
        plt.legend(['logloss'], loc='upper left')
        plt.show()



# Convert timestamp to seconds
def timeToSec(t):
    total_sec = float(t.milliseconds)/1000
    total_sec += t.seconds
    total_sec += t.minutes*60
    total_sec += t.hours*60*60
    return total_sec


# Return timestamp from cell position
def timeToPos(t, freq=Media.FREQ, hop_len=Media.HOP_LEN):
    return round(timeToSec(t)/(hop_len/freq))
