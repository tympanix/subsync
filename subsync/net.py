import os
import tensorflow as tf
import numpy as np

class NeuralNet:
    """
    NeuralNet provides a prediction model for predicting speech using
    Mel-frequency cepstral coefficients (MFCC) data
    """

    DIR = os.path.dirname(os.path.realpath(__file__))

    def __init__(self):
        model = os.path.join(NeuralNet.DIR, 'subsync.pb')
        self.graph = self.load_graph(model)
        self.input = self.graph.get_tensor_by_name('subsync/mfcc_in:0')
        self.output = self.graph.get_tensor_by_name('subsync/speech_out:0')


    def summary(self):
        for op in self.graph.get_operations():
            print(op.name)


    def load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="subsync",
                producer_op_list=None
            )
        return graph


    def predict(self, mfcc):
        print("Predicting values...")
        with tf.Session(graph=self.graph) as sess:
            return sess.run(self.output, feed_dict={self.input: mfcc})
