import argparse

def main(args):
    from media import Media

    media = [Media(m) for m in args]

    for m in media:
        print("Subtitles for", m.filepath)
        for s in m.subtitles():
            print(" -", s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('media', metavar='MEDIA', type=str, nargs='+',
        help='media for which to synchronize subtitles')

    args = parser.parse_args()

    main(args.media)
