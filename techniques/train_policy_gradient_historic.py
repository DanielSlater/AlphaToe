import collections
import functools
import os
import random

import numpy as np
import tensorflow as tf

from common.network_helpers import get_stochastic_network_move, load_network, save_network


def train_policy_gradients_vs_historic(game_spec, create_network, network_file_path,
                                       save_network_file_path=None,
                                       number_of_historic_networks=8,
                                       save_historic_every=10000,
                                       historic_network_base_path='historic_network',
                                       number_of_games=100000,
                                       print_results_every=1000,
                                       learn_rate=1e-4,
                                       batch_size=100):
    """Train a network against itself and over time store new version of itself to play against.

    Args:
        historic_network_base_path (str): Bast path to save new historic networks to a number for the network "slot" is
            appended to the end of this string.
        save_historic_every (int): We save a version of the learning network into one of the historic network
            "slots" every x number of games. We have number_of_historic_networks "slots"
        number_of_historic_networks (int): We keep this many old networks to play against
        save_network_file_path (str): Optionally specifiy a path to use for saving the network, if unset then
            the network_file_path param is used.
        game_spec (games.base_game_spec.BaseGameSpec): The game we are playing
        create_network (->(input_layer : tf.placeholder, output_layer : tf.placeholder, variables : [tf.Variable])):
            Method that creates the network we will train.
        network_file_path (str): path to the file with weights we want to load for this network
        number_of_games (int): number of games to play before stopping
        print_results_every (int): Prints results to std out every x games, also saves the network
        learn_rate (float):
        batch_size (int):

    Returns:
        [tf.Vaiables] : trained variables used in the final network
    """
    input_layer, output_layer, variables = create_network()

    reward_placeholder = tf.placeholder("float", shape=(None,))
    actual_move_placeholder = tf.placeholder("float", shape=(None, game_spec.board_squares()))
    policy_gradient = tf.reduce_sum(tf.reshape(reward_placeholder, (-1, 1)) * actual_move_placeholder * output_layer)
    train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(-policy_gradient)

    current_historical_index = 0
    historical_networks = []

    mini_batch_board_states, mini_batch_moves, mini_batch_rewards = [], [], []
    results = collections.deque(maxlen=print_results_every)

    for _ in range(number_of_historic_networks):
        historical_input_layer, historical_output_layer, historical_variables = create_network()
        historical_networks.append((historical_input_layer, historical_output_layer, historical_variables))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        def make_move_historical(histoical_network_index, board_state, side):
            net = historical_networks[histoical_network_index]
            move = get_stochastic_network_move(session, net[0], net[1], board_state, side,
                                               valid_only=True, game_spec=game_spec)
            return game_spec.flat_move_to_tuple(move.argmax())

        def make_training_move(board_state, side):
            mini_batch_board_states.append(np.ravel(board_state) * side)
            move = get_stochastic_network_move(session, input_layer, output_layer, board_state, side,
                                               valid_only=True, game_spec=game_spec)
            mini_batch_moves.append(move)
            return game_spec.flat_move_to_tuple(move.argmax())

        if os.path.isfile(network_file_path):
            print("loading pre existing weights")
            load_network(session, variables, network_file_path)
        else:
            print("could not find previous weights so initialising randomly")

        for i in range(number_of_historic_networks):
            if os.path.isfile(historic_network_base_path + str(i) + '.p'):
                load_network(session, historical_networks[i][2], historic_network_base_path + str(i) + '.p')
            elif os.path.isfile(network_file_path):
                # if we can't load a historical file use the current network weights
                load_network(session, historical_networks[i][2], network_file_path)

        for episode_number in range(1, number_of_games):
            opponent_index = random.randint(0, number_of_historic_networks - 1)
            make_move_historical_for_index = functools.partial(make_move_historical, opponent_index)

            # randomize if going first or second
            if bool(random.getrandbits(1)):
                reward = game_spec.play_game(make_training_move, make_move_historical_for_index)
            else:
                reward = -game_spec.play_game(make_move_historical_for_index, make_training_move)

            results.append(reward)

            # we scale here so winning quickly is better winning slowly and loosing slowly better than loosing quick
            last_game_length = len(mini_batch_board_states) - len(mini_batch_rewards)

            reward /= float(last_game_length)

            mini_batch_rewards += ([reward] * last_game_length)

            episode_number += 1

            if episode_number % batch_size == 0:
                normalized_rewards = mini_batch_rewards - np.mean(mini_batch_rewards)
                rewards_std = np.std(normalized_rewards)
                if rewards_std != 0:
                    normalized_rewards /= rewards_std
                else:
                    print("warning: got mini batch std of 0.")

                np_mini_batch_board_states = np.array(mini_batch_board_states) \
                                    .reshape(len(mini_batch_rewards), *input_layer.get_shape().as_list()[1:])

                session.run(train_step, feed_dict={input_layer: np_mini_batch_board_states,
                                                   reward_placeholder: normalized_rewards,
                                                   actual_move_placeholder: mini_batch_moves})

                # clear batches
                del mini_batch_board_states[:]
                del mini_batch_moves[:]
                del mini_batch_rewards[:]

            if episode_number % print_results_every == 0:
                print("episode: %s average result: %s" % (episode_number, np.mean(results)))

            if episode_number % save_historic_every == 0:
                print("saving historical network %s", current_historical_index)
                save_network(session, variables, historic_network_base_path + str(current_historical_index) + '.p')
                load_network(session, historical_networks[current_historical_index][2],
                             historic_network_base_path + str(current_historical_index) + '.p')

                # also save to the main network file
                save_network(session, variables, save_network_file_path or network_file_path)

                current_historical_index += 1
                current_historical_index %= number_of_historic_networks

        # save our final weights
        save_network(session, variables, save_network_file_path or network_file_path)

    return variables