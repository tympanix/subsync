# Subsync
**Synchronize your subtitles using machine learning**

Subsync analyses and processes the sound from your media files and uses machine learning to detect speech. Speech detection is used to shift existing subtitles for a perfect match in audio and text!

## Features
 - [x] Machine learning model for voice activity detection (*not recognition*)
 - [x] Shift subtitle as a whole for best match
 - [x] Sync every sentence in the subtitle individually
 - [ ] Sync using existing matched subtitle in a different laguage

## Dependencies
* ffmpeg (https://www.ffmpeg.org/download.html)

## Installation
```bash
pip install subsync
```

## Help
```
usage: subsync [-h] [--version] [--graph] [-d SECONDS] [-m SECONDS] [-s]
                   [--logfile PATH]
                   MEDIA [MEDIA ...]

positional arguments:
  MEDIA                 media for which to synchronize subtitles

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --graph               show graph for subtitle synchronization (default:
                        False)
  -d SECONDS, --duration SECONDS
                        duration (in seconds) of the sample audio length
                        increases precision but reduces speed (default: 900)
  -m SECONDS, --margin SECONDS
                        the margin in which to search for a subtitle match
                        (default: 12)
  -s, --start           sample audio from the start of the media instad of the
                        middle (default: False)
  -r, --recursive       recurviely sync every sentence in the subtitle
                        (default: False)
  --logfile PATH        path to location of log file for logging application
                        specific information (default: None)
```

## Special thanks
[[1] Automatic Subtitle Synchronization through Machine Learning](https://machinelearnings.co/automatic-subtitle-synchronization-e188a9275617) 
