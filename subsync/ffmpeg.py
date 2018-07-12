import random
import string
import subprocess
import os
import tempfile
import re
import sys
from datetime import timedelta
from subprocess import DEVNULL, STDOUT, PIPE


class Transcode:
    """
    Transcode is a wrapper around the ffmpeg binary used to transcode
    audio from media files.
    """

    def __init__(self, input, binary='ffmpeg', seek=False, start=0, duration=0, channels=2, samplerate=16000, bitrate='160k'):
        if seek and start:
            raise ValueError("Can't both supply seek and start argument in transcode")
        self.input = input
        self.bitrate = bitrate
        self.channels = channels
        self.samplerate = samplerate
        self.binary =  binary
        self.start = start if type(start) is timedelta else timedelta(seconds=start)
        self.duration = duration if type(duration) is timedelta else timedelta(seconds=duration)
        self.length = self.__length()
        if seek:
            self.start = max(timedelta(), self.length/2-self.duration/2)

        self.output = os.path.join(tempfile.gettempdir(), 'subsync_' + randomString() + '.wav')


    def command(self):
        cmd = [self.binary, '-y']
        cmd.extend(('-i', shellquote(self.input)))

        if self.start > timedelta():
            cmd.extend(('-ss', duration_str(self.start)))

        if self.duration > timedelta():
            cmd.extend(('-t', self.duration.seconds))

        cmd.extend(('-ab', self.bitrate))
        cmd.extend(('-ac', self.channels))
        cmd.extend(('-ar', self.samplerate))
        cmd.append('-vn') # no video
        cmd.append(self.output)

        return [str(s) for s in cmd]


    def __length(self):
        cmd = subprocess.Popen(['ffprobe', self.input], stdout=PIPE, stderr=STDOUT)
        duration = [x.decode("utf-8") for x in cmd.stdout.readlines() if b"Duration" in x]
        match = re.search(r'(\d\d):(\d\d):(\d\d)\.(\d\d)', duration[0])
        code = cmd.wait()
        if not match or code != 0:
            raise RuntimeError('Could not call ffprobe:', self.input)
        return timedelta(
            hours=int(match.group(1)),
            minutes=int(match.group(2)),
            seconds=int(match.group(3)),
            milliseconds=int(match.group(4))*100
        )


    def run(self):
        code = subprocess.call(' '.join(self.command()), stderr=DEVNULL, shell=True)
        if code != 0:
            raise RuntimeError('Could not transcode audio:', self.input)


def randomString(len=12):
    allchar = string.ascii_letters + string.digits
    return "".join(random.choice(allchar) for x in range(len))


def duration_str(d):
    hours, remainder = divmod(d.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}.{:06d}'.format(hours, minutes, seconds, d.microseconds)


def shellquote(s):
    if sys.platform == 'win32':
        return "\"" + s.replace("\"", "\\\"") + "\""
    else:
        return "'" + s.replace("'", "'\\''") + "'"
