import os
import librosa
import subprocess
import tempfile

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
    FFMPEG_CMD = "ffmpeg -y -i {0} -ab 160k -ac 2 -ar {2} -vn {1}"

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
                yield f


    def mfcc(self):
        with tempfile.NamedTemporaryFile() as output:
            command = FFMPEG_CMD.format(self.filepath, output.name, FREQ)
            subprocess.call(command, stderr=subprocess.DEVNULL, shell=True)
            y, sr = librosa.load(output.name, sr=FREQ)
            return librosa.feature.mfcc(y=y, sr=sr,
                hop_length=int(Media.HOP_LEN),
                n_mfcc=int(Media.N_MFCC)
            )
