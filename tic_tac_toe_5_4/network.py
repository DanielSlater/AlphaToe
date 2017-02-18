import tensorflow as tf

from common.benchmark import benchmark
from games.tic_tac_toe_x import TicTacToeXGameSpec

tic_tac_toe_5_4_game_spec = TicTacToeXGameSpec(5, 4)


def create_convolutional_network():
    input_layer = tf.input_layer = tf.placeholder("float",
                                                  (None,) + tic_tac_toe_5_4_game_spec.board_dimensions() + (1,))
    CONVOLUTIONS_LAYER_1 = 64
    CONVOLUTIONS_LAYER_2 = 64
    CONVOLUTIONS_LAYER_3 = 64
    CONVOLUTIONS_LAYER_4 = 64
    CONVOLUTIONS_LAYER_5 = 64
    FLAT_SIZE = 5 * 5 * CONVOLUTIONS_LAYER_2
    FLAT_HIDDEN_NODES = 256

    convolution_weights_1 = tf.Variable(tf.truncated_normal([3, 3, 1, CONVOLUTIONS_LAYER_1], stddev=0.01))
    convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[CONVOLUTIONS_LAYER_1]))

    convolution_weights_2 = tf.Variable(
        tf.truncated_normal([3, 3, CONVOLUTIONS_LAYER_1, CONVOLUTIONS_LAYER_2], stddev=0.01))
    convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[CONVOLUTIONS_LAYER_2]))

    convolution_weights_3 = tf.Variable(
        tf.truncated_normal([3, 3, CONVOLUTIONS_LAYER_2, CONVOLUTIONS_LAYER_3], stddev=0.01))
    convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[CONVOLUTIONS_LAYER_3]))

    convolution_weights_4 = tf.Variable(
        tf.truncated_normal([3, 3, CONVOLUTIONS_LAYER_3, CONVOLUTIONS_LAYER_4], stddev=0.01))
    convolution_bias_4 = tf.Variable(tf.constant(0.01, shape=[CONVOLUTIONS_LAYER_4]))

    # convolution_weights_5 = tf.Variable(
    #     tf.truncated_normal([3, 3, CONVOLUTIONS_LAYER_4, CONVOLUTIONS_LAYER_5], stddev=0.01))
    # convolution_bias_5 = tf.Variable(tf.constant(0.01, shape=[CONVOLUTIONS_LAYER_5]))

    # feed_forward_weights_1 = tf.Variable(tf.truncated_normal([FLAT_SIZE, FLAT_HIDDEN_NODES], stddev=0.01))
    # feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[FLAT_HIDDEN_NODES]))

    feed_forward_weights_2 = tf.Variable(
        tf.truncated_normal([FLAT_SIZE, tic_tac_toe_5_4_game_spec.outputs()], stddev=0.01))
    feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[tic_tac_toe_5_4_game_spec.outputs()]))

    hidden_convolutional_layer_1 = tf.nn.relu(
        tf.nn.conv2d(input_layer, convolution_weights_1, strides=[1, 1, 1, 1], padding="SAME") + convolution_bias_1)

    hidden_convolutional_layer_2 = tf.nn.relu(
        tf.nn.conv2d(hidden_convolutional_layer_1, convolution_weights_2, strides=[1, 1, 1, 1],
                     padding="SAME") + convolution_bias_2)

    hidden_convolutional_layer_3 = tf.nn.relu(
        tf.nn.conv2d(hidden_convolutional_layer_2, convolution_weights_3, strides=[1, 1, 1, 1],
                     padding="SAME") + convolution_bias_3)

    hidden_convolutional_layer_4 = tf.nn.relu(
        tf.nn.conv2d(hidden_convolutional_layer_3, convolution_weights_4, strides=[1, 1, 1, 1],
                     padding="SAME") + convolution_bias_4)

    # hidden_convolutional_layer_5 = tf.nn.relu(
    #     tf.nn.conv2d(hidden_convolutional_layer_4, convolution_weights_5, strides=[1, 1, 1, 1],
    #                  padding="SAME") + convolution_bias_5)

    hidden_convolutional_layer_3_flat = tf.reshape(hidden_convolutional_layer_4, [-1, FLAT_SIZE])

    # final_hidden_activations = tf.nn.relu(
    #     tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

    output_layer = tf.nn.softmax(tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_2) + feed_forward_bias_2)

    return input_layer, output_layer, [convolution_weights_1, convolution_bias_1,
                                       convolution_weights_2, convolution_bias_2,
                                       convolution_weights_3, convolution_bias_3,
                                       convolution_weights_4, convolution_bias_4,
                                       # convolution_weights_5, convolution_bias_5,
                                       # feed_forward_weights_1, feed_forward_bias_1,
                                       feed_forward_weights_2, feed_forward_bias_2]

file_path = 'convolutional_net_5_4_l_c_4_f_1_other_fresh.p'

benchmark(tic_tac_toe_5_4_game_spec, file_path, create_convolutional_network)