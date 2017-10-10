import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

model_filename ='./out/opt_mnist_convnet.pb'

with tf.Session() as sess:
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='/tmp/tensorflow/android'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)