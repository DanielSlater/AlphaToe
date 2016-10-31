import collections
import os
import random

import numpy as np
import tensorflow as tf

from common.network_helpers import load_network, get_stochastic_network_move, save_network


def train_policy_gradients_vs_random(game_spec,
                                     create_network,
                                     network_file_path,
                                     number_of_games=10000,
                                     print_results_every=1000,
                                     learn_rate=1e-4,
                                     batch_size=100,
                                     randomize_first_player=True):
    """Train a network using policy gradients

    Args:
        randomize_first_player (bool): If True we alternate between being the first and second player
        game_spec (games.base_game_spec.BaseGameSpec):
        create_network (->(input_layer, output_layer, variables)):
        network_file_path (str):
        number_of_games (int):
        print_results_every (int):
        learn_rate (float):
        batch_size (int):

    Returns:

    """
    reward_placeholder = tf.placeholder("float", shape=(None,))
    actual_move_placeholder = tf.placeholder("float", shape=(None, game_spec.outputs()))

    input_layer, output_layer, variables = create_network()

    policy_gradient = tf.reduce_sum(tf.reshape(reward_placeholder, (-1, 1)) * actual_move_placeholder * output_layer)
    train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(-policy_gradient)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        if network_file_path and os.path.isfile(network_file_path):
            print("loading pre-existing network")
            load_network(session, variables, network_file_path)

        mini_batch_board_states, mini_batch_moves, mini_batch_rewards = [], [], []
        results = collections.deque(maxlen=print_results_every)

        def make_training_move(board_state, side):
            mini_batch_board_states.append(np.ravel(board_state) * side)
            move = get_stochastic_network_move(session, input_layer, output_layer, board_state, side)
            mini_batch_moves.append(move)
            return game_spec.flat_move_to_tuple(move.argmax())

        for episode_number in range(1, number_of_games):
            # randomize if going first or second
            if (not randomize_first_player) or bool(random.getrandbits(1)):
                reward = game_spec.play_game(make_training_move, game_spec.get_random_player_func())
            else:
                reward = -game_spec.play_game(game_spec.get_random_player_func(), make_training_move)

            results.append(reward)

            # we scale here so winning quickly is better winning slowly and loosing slowly better than loosing quick
            last_game_length = len(mini_batch_board_states) - len(mini_batch_rewards)

            reward /= float(last_game_length)

            mini_batch_rewards += ([reward] * last_game_length)

            if episode_number % batch_size == 0:
                normalized_rewards = mini_batch_rewards - np.mean(mini_batch_rewards)

                rewards_std = np.std(normalized_rewards)
                if rewards_std != 0:
                    normalized_rewards /= rewards_std
                else:
                    print("warning: got mini batch std of 0.")

                np_mini_batch_board_states = np.array(mini_batch_board_states)\
                    .reshape(len(mini_batch_rewards), *input_layer.get_shape().as_list()[1:])

                session.run(train_step, feed_dict={input_layer: np_mini_batch_board_states,
                                                   reward_placeholder: normalized_rewards,
                                                   actual_move_placeholder: mini_batch_moves})

                # clear batches
                del mini_batch_board_states[:]
                del mini_batch_moves[:]
                del mini_batch_rewards[:]

            if episode_number % print_results_every == 0:
                print("episode: %s win_rate: %s" % (episode_number, _win_rate(print_results_every, results)))
                if network_file_path:
                    save_network(session, variables, network_file_path)

        if network_file_path:
            save_network(session, variables, network_file_path)

        return variables, _win_rate(print_results_every, results)


def _win_rate(print_results_every, results):
    return 0.5 + sum(results) / (print_results_every * 2.)