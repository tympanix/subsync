import argparse

def main(args):
    from media import Media
    media = [Media(m) for m in args]

    from net import NeuralNet
    model = NeuralNet()

    for m in media:
        m.mfcc()
        for s in m.subtitles():
            s.sync(model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('media', metavar='MEDIA', type=str, nargs='+',
        help='media for which to synchronize subtitles')

    args = parser.parse_args()

    main(args.media)
