import argparse
import os

from .log import logger, init_logger
from .version import __version__


def run():
    parser = argparse.ArgumentParser(
        description="""
        Utility to synchronize a single srt-fromatted subtitles file with a video file.

        The srt file is modified in-place so this utility must have read+write access to it

        Example:
        subsync --media video.mp4 --srt video.en.srt -r
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--media",
        dest="media",
        required=True,
        type=str,
        metavar="PATH",
        help="Path to video file for which to synchronize subtitles (mkv, mp4, wmv, avi, flv)",
    )
    parser.add_argument(
        "--srt",
        dest="srt",
        required=True,
        type=str,
        metavar="PATH",
        help="Path to subtitles file to synchronize (srt format only)",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s {}".format(__version__)
    )
    parser.add_argument(
        "--graph",
        dest="graph",
        action="store_true",
        help="show graph for subtitle synchronization",
    )
    parser.add_argument(
        "-d",
        "--duration",
        dest="duration",
        type=int,
        metavar="SECONDS",
        default=60 * 15,
        help="duration (in seconds) of the sample audio length increases precision but reduces speed",
    )
    parser.add_argument(
        "-m",
        "--margin",
        dest="margin",
        type=int,
        metavar="SECONDS",
        default=12,
        help="the margin in which to search for a subtitle match",
    )
    parser.add_argument(
        "-s",
        "--start",
        dest="start",
        action="store_true",
        help="sample audio from the start of the media instad of the middle",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        action="store_true",
        help="recurviely sync every sentence in the subtitle",
    )
    parser.add_argument(
        "--logfile",
        dest="logfile",
        type=str,
        metavar="PATH",
        help="path to location of log file for logging application specific information",
    )

    args = parser.parse_args()

    if args.logfile:
        init_logger(args.logfile)

    from .media import Media

    if not os.path.exists(args.media):
        raise ValueError(f"{args.media} does not exist")
    if not os.path.exists(args.srt):
        raise ValueError(f"{args.srt} does not exist")

    media = Media(filepath=args.media, subtitles=[args.srt])

    from .net import NeuralNet

    model = NeuralNet()

    if args.recursive:
        media.mfcc(duration=0, seek=False)
    else:
        media.mfcc(duration=args.duration, seek=not args.start)
    for s in media.subtitles():
        if args.recursive:
            s.sync_all(model, plot=args.graph, margin=args.margin)
        else:
            s.sync(model, plot=args.graph, margin=args.margin)
