"""
After using reinforcement learning to train a network, e.g. policy_gradient.py, to play a game well. We then want to
learn to estimate weather that network would win, lose or draw from a given position.

Alpha Go used a database of real positions to get it's predictions from, we don't have that for tic-tac-toe so instead
we generate some random game positions and train off of the results we get playing from those.
"""
import os
import random

import numpy as np
import tensorflow as tf

from common.network_helpers import create_network, load_network, save_network, \
    get_deterministic_network_move
from games.tic_tac_toe import TicTacToeGameSpec

HIDDEN_NODES_VALUE = (100, 100, 100)
HIDDEN_NODES_REINFORCEMENT = (100, 100, 100)
BATCH_SIZE = 100  # every how many games to do a parameter update?
LEARN_RATE = 1e-4
REINFORCEMENT_NETWORK_PATH = 'current_network.p'
VALUE_NETWORK_PATH = 'value_netowrk.p'
TRAIN_SAMPLES = 10000
TEST_SAMPLES = 10000

# to play a different game change this to another spec, e.g TicTacToeXGameSpec or ConnectXGameSpec
game_spec = TicTacToeGameSpec()

NUMBER_RANDOM_RANGE = (1, game_spec.board_squares() * 0.8)


# it would be good to have real board positions, but failing that just generate random ones
def generate_random_board_position():
    while True:
        board_state = game_spec.new_board()
        number_moves = random.randint(*NUMBER_RANDOM_RANGE)
        side = 1
        for _ in range(number_moves):
            board_state = game_spec.apply_move(board_state, random.choice(list(game_spec.available_moves(board_state))),
                                               side)
            if game_spec.has_winner(board_state) != 0:
                # start again if we hit an already winning position
                continue

            side = -side
        return board_state


reinforcement_input_layer, reinforcement_output_layer, reinforcement_variables = create_network(
    game_spec.board_squares(),
    HIDDEN_NODES_REINFORCEMENT,
    game_spec.outputs())

value_input_layer, value_output_layer, value_variables = create_network(game_spec.board_squares(), HIDDEN_NODES_VALUE,
                                                                        output_nodes=1, output_softmax=False)

target_placeholder = tf.placeholder("float", (None, 1))
error = tf.reduce_sum(tf.square(target_placeholder - value_output_layer))

train_step = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(error)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    load_network(session, reinforcement_variables, REINFORCEMENT_NETWORK_PATH)

    if os.path.isfile(VALUE_NETWORK_PATH):
        print("loading previous version of value network")
        load_network(session, value_variables, VALUE_NETWORK_PATH)


    def make_move(board_state, side):
        move = get_deterministic_network_move(session, reinforcement_input_layer, reinforcement_output_layer,
                                              board_state, side)

        return game_spec.flat_move_to_tuple(np.argmax(move))


    board_states_training = {}
    board_states_test = []
    episode_number = 0

    while len(board_states_training) < TRAIN_SAMPLES + TEST_SAMPLES:
        board_state = generate_random_board_position()
        board_state_flat = tuple(np.ravel(board_state))

        # only accept the board_state if not already in the dict
        if board_state_flat not in board_states_training:
            result = game_spec.play_game(make_move, make_move, board_state=board_state)
            board_states_training[board_state_flat] = float(result)

    # take a random selection from training into a test set
    for _ in range(TEST_SAMPLES):
        sample = random.choice(board_states_training.keys())
        board_states_test.append((sample, board_states_training[sample]))
        del board_states_training[sample]

    board_states_training = list(board_states_training.iteritems())

    test_error = session.run(error, feed_dict={value_input_layer: [x[0] for x in board_states_test],
                                               target_placeholder: [[x[1]] for x in board_states_test]})

    while True:
        np.random.shuffle(board_states_training)
        train_error = 0

        for start_index in range(0, len(board_states_training) - BATCH_SIZE + 1, BATCH_SIZE):
            mini_batch = board_states_training[start_index:start_index + BATCH_SIZE]

            batch_error, _ = session.run([error, train_step],
                                         feed_dict={value_input_layer: [x[0] for x in mini_batch],
                                                    target_placeholder: [[x[1]] for x in mini_batch]})
            train_error += batch_error

        new_test_error = session.run(error, feed_dict={value_input_layer: [x[0] for x in board_states_test],
                                                       target_placeholder: [[x[1]] for x in board_states_test]})

        print("episode: %s train_error: %s test_error: %s" % (episode_number, train_error, test_error))

        if new_test_error > test_error:
            print("train error went up, stopping training")
            break

        test_error = new_test_error
        episode_number += 1

    save_network(session, value_variables, VALUE_NETWORK_PATH)
