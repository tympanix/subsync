# Update

## Updated version to run with tensorflow 2.0**

The previous version would not install, so I had to make some modifications to this outstanding project.
Tested on M1 Silicon Ventura.
Use of Conda (Miniconda) is strongly recommended.
Only minor tweaks were made, so bugs are expected.
Check the installation section.

# Subsync

**Synchronize your subtitles using machine learning**

Subsync analyses and processes the sound from your media files and uses machine learning to detect speech. Speech detection is used to shift existing subtitles for a perfect match in audio and text!

## Features

- [x] Machine learning model for voice activity detection (*not recognition*)
- [x] Shift subtitle as a whole for best match
- [x] Sync every sentence in the subtitle individually
- [ ] Sync using existing matched subtitle in a different laguage

## Dependencies

- ffmpeg (<https://www.ffmpeg.org/download.html>)

## Installation

1. Install miniconda (<https://docs.conda.io/en/latest/miniconda.html>).
2. Create and activate a subsync environment:

```bash
conda create --name subsync-env
conda activate subsync-env
```

3. Install the package:

```bash
pip install git+https://github.com/StanislavAlexandrov/subsync
```

## Usage

1. The video file and the subtitle file must have the same name. For instance myShow_s01e01.mkv and myShow_s01e01.srt
2. Run the following command:

```bash
subsync -r myShow_s01e01.srt
```

3. The subtitle file will be overwritten with the synchronized subtitles! As of now, there is no indication that the script has run successfully.

## Options

```bash
subsync -h
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
