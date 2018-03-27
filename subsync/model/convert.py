# https://github.com/bitbionic/keras-to-tensorflow

import os
import os.path as osp
import argparse

import tensorflow as tf

from keras.models import load_model
from keras import backend as K

def convertGraph(modelPath, output, outPath):
    '''
    Converts an HD5F file to a .pb file for use with Tensorflow.
    Args:
        modelPath (str): path to the .h5 file
           output (str): name of the referenced output
          outPath (str): path to the output .pb file
    Returns:
        None
    '''

    dir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(dir, os.path.dirname(outPath))
    name = os.path.basename(outPath)
    basename, ext = os.path.splitext(name)

    #NOTE: If using Python > 3.2, this could be replaced with os.makedirs( name, exist_ok=True )
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    K.set_learning_phase(0)

    net_model = load_model(modelPath)

    # Alias the outputs in the model - this sometimes makes them easier to access in TF
    tf.identity(net_model.output, name=output)

    sess = K.get_session()

    net_model.summary()

    # Write the graph in human readable
    f = '{}.reference.pb.ascii'.format(basename)
    tf.train.write_graph(sess.graph.as_graph_def(), outdir, f, as_text=True)
    print('Saved the graph definition in ascii format at: ', osp.join(outdir, f))

    # Write the graph in binary .pb file
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output])
    graph_io.write_graph(constant_graph, outdir, name, as_text=False)
    print('Saved the constant graph (ready for inference) at: ', outPath)


if __name__ == '__main__':
    convertGraph('out/ann.hdf5', 'speech_out', 'out/subsync.pb')
