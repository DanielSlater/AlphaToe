import os
import random

import numpy as np
import tensorflow as tf

from common.network_helpers import create_network, load_network, get_deterministic_network_move, save_network


# it would be good to have real board positions, but failing that just generate random ones
def _generate_random_board_position(game_spec, random_move_range):
    while True:
        board_state = game_spec.new_board()
        number_moves = random.randint(*random_move_range)
        side = 1
        for _ in range(number_moves):
            board_state = game_spec.apply_move(board_state, random.choice(list(game_spec.available_moves(board_state))),
                                               side)
            if game_spec.has_winner(board_state) != 0:
                # start again if we hit an already winning position
                continue

            side = -side
        return board_state


def train_value_network(game_spec, hidden_nodes_reinforcement, reinforcement_network_file_path,
                        hidden_nodes_value, value_network_file_path,
                        learn_rate=1e-4,
                        batch_size=100,
                        train_samples=10000,
                        test_samples=8000):
    reinforcement_input_layer, reinforcement_output_layer, reinforcement_variables = create_network(
        game_spec.board_squares(),
        hidden_nodes_reinforcement,
        game_spec.outputs())

    value_input_layer, value_output_layer, value_variables = create_network(game_spec.board_squares(),
                                                                            hidden_nodes_value,
                                                                            output_nodes=1, output_softmax=False)

    target_placeholder = tf.placeholder("float", (None, 1))
    error = tf.reduce_sum(tf.square(target_placeholder - value_output_layer))

    train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(error)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        load_network(session, reinforcement_variables, reinforcement_network_file_path)

        if os.path.isfile(value_network_file_path):
            print("loading previous version of value network")
            load_network(session, value_variables, value_network_file_path)

        def make_move(board_state, side):
            move = get_deterministic_network_move(session, reinforcement_input_layer, reinforcement_output_layer,
                                                  board_state, side)

            return game_spec.flat_move_to_tuple(np.argmax(move))

        board_states_training = {}
        board_states_test = []
        episode_number = 0

        while len(board_states_training) < train_samples + test_samples:
            board_state = _generate_random_board_position(game_spec, (1, game_spec.board_squares() * 0.8))
            board_state_flat = tuple(np.ravel(board_state))

            # only accept the board_state if not already in the dict
            if board_state_flat not in board_states_training:
                result = game_spec.play_game(make_move, make_move, board_state=board_state)
                board_states_training[board_state_flat] = float(result)

        # take a random selection from training into a test set
        for _ in range(test_samples):
            sample = random.choice(board_states_training.keys())
            board_states_test.append((sample, board_states_training[sample]))
            del board_states_training[sample]

        board_states_training = list(board_states_training.iteritems())

        test_error = session.run(error, feed_dict={value_input_layer: [x[0] for x in board_states_test],
                                                   target_placeholder: [[x[1]] for x in board_states_test]})

        while True:
            np.random.shuffle(board_states_training)
            train_error = 0

            for start_index in range(0, len(board_states_training) - batch_size + 1, batch_size):
                mini_batch = board_states_training[start_index:start_index + batch_size]

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

        save_network(session, value_variables, value_network_file_path)
