import numpy as np
import tensorflow as tf


def load_games():
    """If we had a database of games this would load and return it...

    Returns:

    """
    raise Exception("If we had a database of tic-tac-toe games this would load them")


HIDDEN_NODES = (100, 100, 100)  # number of hidden layer neurons
INPUT_NODES = 3 * 3  # board size
BATCH_SIZE = 100  # every how many games to do a parameter update?
LEARN_RATE = 1e-4
OUTPUT_NODES = INPUT_NODES
PRINT_RESULTS_EVERY_X = 1000  # every how many games to print the results

input_placeholder = tf.placeholder("float", shape=(None, INPUT_NODES))
actual_move_placeholder = tf.placeholder("float", shape=(None, OUTPUT_NODES))

hidden_weights_1 = tf.Variable(tf.truncated_normal((INPUT_NODES, HIDDEN_NODES[0]), stddev=1. / np.sqrt(INPUT_NODES)))
hidden_weights_2 = tf.Variable(
    tf.truncated_normal((HIDDEN_NODES[0], HIDDEN_NODES[1]), stddev=1. / np.sqrt(HIDDEN_NODES[0])))
hidden_weights_3 = tf.Variable(
    tf.truncated_normal((HIDDEN_NODES[1], HIDDEN_NODES[2]), stddev=1. / np.sqrt(HIDDEN_NODES[1])))
output_weights = tf.Variable(tf.truncated_normal((HIDDEN_NODES[-1], OUTPUT_NODES), stddev=1. / np.sqrt(OUTPUT_NODES)))

hidden_layer_1 = tf.nn.relu(
    tf.matmul(input_placeholder, hidden_weights_1) + tf.Variable(tf.constant(0.01, shape=(HIDDEN_NODES[0],))))
hidden_layer_2 = tf.nn.relu(
    tf.matmul(hidden_layer_1, hidden_weights_2) + tf.Variable(tf.constant(0.01, shape=(HIDDEN_NODES[1],))))
hidden_layer_3 = tf.nn.relu(
    tf.matmul(hidden_layer_2, hidden_weights_3) + tf.Variable(tf.constant(0.01, shape=(HIDDEN_NODES[2],))))
output_layer = tf.nn.softmax(
    tf.matmul(hidden_layer_3, output_weights) + tf.Variable(tf.constant(0.01, shape=(OUTPUT_NODES,))))

error = tf.reduce_sum(tf.square(actual_move_placeholder - output_layer))
train_step = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(error)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

episode_number = 1

positions_train, positions_test = load_games()

test_error = sess.run(error, feed_dict={input_placeholder: [x[0] for x in positions_test]})

while True:
    np.random.shuffle(positions_train)
    train_error = 0

    for start_index in range(0, positions_train.shape[0] - BATCH_SIZE + 1, BATCH_SIZE):
        mini_batch = positions_train[start_index:start_index + BATCH_SIZE]

        error, _ = sess.run(error, train_step, feed_dict={input_placeholder: [x[0] for x in mini_batch],
                                                          actual_move_placeholder: [x[1] for x in mini_batch]})
        train_error += error

    new_test_error = sess.run(error, feed_dict={input_placeholder: [x[0] for x in positions_test]})

    print("episode: %s train_error: %s test_error: %s" % (episode_number, train_error, test_error))

    if new_test_error > test_error:
        print("train error went up, stopping training")

    test_error = new_test_error
    episode_number += 1
