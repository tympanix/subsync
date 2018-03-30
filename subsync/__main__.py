import argparse

def main(args):
    from media import Media
    media = [Media(m) for m in args.media]

    from net import NeuralNet
    model = NeuralNet()

    for m in media:
        m.mfcc()
        for s in m.subtitles():
            s.sync(model, plot=args.graph)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('media', metavar='MEDIA', type=str, nargs='+',
        help='media for which to synchronize subtitles')
    parser.add_argument('--graph', dest="graph", action='store_true', help='show graph for subtitle synchronization')

    args = parser.parse_args()

    main(args)
