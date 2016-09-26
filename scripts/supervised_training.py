import os

import numpy as np
import tensorflow as tf

from games.tic_tac_toe_x import TicTacToeXGameSpec
from network_helpers import create_network, save_network, load_network


def load_games():
    """If we had a database of games this would load and return it...

    Returns:

    """
    raise Exception("If we had a database of tic-tac-toe games this would load them")


HIDDEN_NODES = (100, 80, 60, 40)  # number of hidden layer neurons
BATCH_SIZE = 100  # every how many games to do a parameter update?
LEARN_RATE = 1e-4
NETWORK_FILE_PATH = 'current_network.p'
game_spec = TicTacToeXGameSpec(5, 4)

input_layer, output_layer, variables = create_network(game_spec.board_squares(), HIDDEN_NODES,
                                                      output_nodes=game_spec.outputs())
actual_move_placeholder = tf.placeholder("float", (None, game_spec.outputs()))

error = tf.reduce_sum(tf.square(actual_move_placeholder - output_layer))
train_step = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(error)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    if os.path.isfile(NETWORK_FILE_PATH):
        print("loading existing network")
        load_network(session, variables, NETWORK_FILE_PATH)

    episode_number = 1

    positions_train, positions_test = load_games()

    test_error = session.run(error, feed_dict={input_layer: [x[0] for x in positions_test],
                                               actual_move_placeholder: [[x[1]] for x in positions_test]})

    while True:
        np.random.shuffle(positions_train)
        train_error = 0

        for start_index in range(0, positions_train.shape[0] - BATCH_SIZE + 1, BATCH_SIZE):
            mini_batch = positions_train[start_index:start_index + BATCH_SIZE]

            batch_error, _ = session.run([error, train_step],
                                         feed_dict={input_layer: [x[0] for x in mini_batch],
                                                    actual_move_placeholder: [[x[1]] for x in mini_batch]})
            train_error += batch_error

        new_test_error = session.run(error, feed_dict={input_layer: [x[0] for x in positions_test],
                                                       actual_move_placeholder: [[x[1]] for x in positions_test]})

        print("episode: %s train_error: %s test_error: %s" % (episode_number, train_error, test_error))

        if new_test_error > test_error:
            print("train error went up, stopping training")
            break

        test_error = new_test_error
        episode_number += 1

    save_network(session, variables, NETWORK_FILE_PATH)
