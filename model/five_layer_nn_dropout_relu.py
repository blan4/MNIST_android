import math
import os
import os.path as path

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = 'mnist_convnet'


def nn_layer(X, input_dim, output_dim, layer_name, name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
        with tf.name_scope('biases'):
            B = tf.Variable(tf.zeros(output_dim))
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(X, W) + B
            Y = act(preactivate, name=name)

        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', B)
        tf.summary.histogram('pre_activations', preactivate)
        tf.summary.histogram('activations', Y)

        return Y, preactivate


def build_model():
    layer_sizes = [28 * 28, 200, 100, 60, 30, 10]

    with tf.name_scope('X'):
        X = tf.placeholder(tf.float32, [None, 784], name='X')
        x_image = tf.reshape(X, [-1, 28, 28, 1])
        tf.summary.image('input_image', x_image, 10)

    Y1 = nn_layer(X, layer_sizes[0], layer_sizes[1], "first", 'Y1')[0]
    Y2 = nn_layer(Y1, layer_sizes[1], layer_sizes[2], "second", 'Y2')[0]
    Y3 = nn_layer(Y2, layer_sizes[2], layer_sizes[3], "third", 'Y3')[0]
    Y4 = nn_layer(Y3, layer_sizes[3], layer_sizes[4], "fourth", 'Y4')[0]
    Y, Ylogits = nn_layer(Y4, layer_sizes[4], layer_sizes[5], 'fifth', 'Y', tf.nn.softmax)

    with tf.name_scope('Y_'):
        Y_ = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope('xentropy'):
        # Функция потерь H = Sum(Y_ * log(Y))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
        cross_entropy = tf.reduce_mean(cross_entropy) * 100
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # Доля верных ответов найденных в наборе
            is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
        with tf.name_scope('xentropy_mean'):
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('train'):
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # Оптимизируем функцию потерь меотодом градиентного спуска
        # 0.003 - это шаг градиента, гиперпараметр
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Минимизируем потери
        train_step = optimizer.minimize(cross_entropy)

    return X, Y_, learning_rate, train_step, accuracy


def train(X, Y_, learning_rate, train_step, accuracy, saver, epoches=1000, batch_size=100):
    print("Start model")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=True)

    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        tf.train.write_graph(sess.graph_def, 'out',
                             MODEL_NAME + '.pbtxt', True)

        merged = tf.summary.merge_all()  # Merge all the summaries and write them out to
        writer = tf.summary.FileWriter("/tmp/tensorflow/five_layer_nn_dropout_relu", sess.graph)

        for i in range(epoches):
            # загружаем набор изображений и меток классов
            batch_X, batch_Y = mnist.train.next_batch(batch_size)

            lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)
            train_data = {X: batch_X, Y_: batch_Y, learning_rate: lr}

            # train
            sess.run(train_step, feed_dict=train_data)

            if i % 10 == 0:
                test_data = {X: mnist.test.images, Y_: mnist.test.labels}
                summary, a = sess.run([merged, accuracy], feed_dict=test_data)
                writer.add_summary(summary, i)
                if i % 200 == 0:
                    print("Test: {}".format(a))

        saver.save(sess, 'out/' + MODEL_NAME + '.chkp')

        writer.close()

    print("Training finished")


def export_model(input_node_names, output_node_name):
    if not path.exists('out'):
        os.mkdir('out')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
                              'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
                              "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


def main():
    print("MNIST single layer NN")
    tf.set_random_seed(0)
    tf.reset_default_graph()
    X, Y_, learning_rate, train_step, accuracy = build_model()
    saver = tf.train.Saver()
    train(X, Y_, learning_rate, train_step, accuracy, saver, epoches=100)
    export_model(['X'], 'Y')


if __name__ == '__main__':
    main()
