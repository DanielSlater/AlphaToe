"""
This is the same as the policy_gradient.py network except that instead of playing against a random opponent. It plays
against previous versions of itself. It is first created with the weights from the "current_network.p" file, if no file
is found there random weights are used. It then creates a series of copies of itself and plays against them.
After "SAVE_HISTORICAL_NETWORK_EVERY" games, it saves it's current weights into the weights of one of the historical
networks. Over time the main network and the historical networks should improve.
"""
import collections
import functools
import os
import random

import numpy as np
import tensorflow as tf

from games.tic_tac_toe import TicTacToeGameSpec
from network_helpers import create_network, load_network, get_stochastic_network_move, \
    save_network

NUMBER_OF_HISTORICAL_COPIES_TO_KEEP = 8
NUMBER_OF_GAMES_TO_PLAY = 1000000
MINI_BATCH_SIZE = 100
SAVE_HISTORICAL_NETWORK_EVERY = 100000
STARTING_NETWORK_WEIGHTS = 'current_network.p'
BASE_HISTORICAL_NETWORK_PATH = 'historical_network_'
HIDDEN_NODES = (100, 80, 60, 40)
PRINT_RESULTS_EVERY_X = 500
LEARN_RATE = 1e-4
game_spec = TicTacToeGameSpec()

input_layer, output_layer, variables = create_network(game_spec.board_squares(), HIDDEN_NODES,
                                                      output_nodes=game_spec.outputs())

reward_placeholder = tf.placeholder("float", shape=(None,))
actual_move_placeholder = tf.placeholder("float", shape=(None, game_spec.board_squares()))
policy_gradient = tf.reduce_sum(tf.reshape(reward_placeholder, (-1, 1)) * actual_move_placeholder * output_layer)
train_step = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(-policy_gradient)

current_historical_index = 0
historical_networks = []

mini_batch_board_states, mini_batch_moves, mini_batch_rewards = [], [], []
results = collections.deque(maxlen=PRINT_RESULTS_EVERY_X)

for _ in range(NUMBER_OF_HISTORICAL_COPIES_TO_KEEP):
    historical_input_layer, historical_output_layer, historical_variables = create_network(game_spec.board_squares(),
                                                                                           HIDDEN_NODES)
    historical_networks.append((historical_input_layer, historical_output_layer, historical_variables))

with tf.Session() as session:
    session.run(tf.initialize_all_variables())


    def make_move_historical(histoical_network_index, board_state, side):
        net = historical_networks[histoical_network_index]
        move = get_stochastic_network_move(session, net[0], net[1], board_state, side)
        return game_spec.flat_move_to_tuple(move.argmax())


    def make_training_move(board_state, side):
        mini_batch_board_states.append(np.ravel(board_state) * side)
        move = get_stochastic_network_move(session, input_layer, output_layer, board_state, side)
        mini_batch_moves.append(move)
        return game_spec.flat_move_to_tuple(move.argmax())


    if os.path.isfile(STARTING_NETWORK_WEIGHTS):
        print("loading pre existing weights")
        load_network(session, variables, STARTING_NETWORK_WEIGHTS)
    else:
        print("could not find previous weights so initialising randomly")

    for i in range(NUMBER_OF_HISTORICAL_COPIES_TO_KEEP):
        if os.path.isfile(BASE_HISTORICAL_NETWORK_PATH + str(i) + '.p'):
            load_network(session, historical_networks[i][2], BASE_HISTORICAL_NETWORK_PATH + str(i) + '.p')
        elif os.path.isfile(STARTING_NETWORK_WEIGHTS):
            # if we can't load a historical file use the current network weights
            load_network(session, historical_networks[i][2], STARTING_NETWORK_WEIGHTS)

    for episode_number in range(1, NUMBER_OF_GAMES_TO_PLAY):
        opponent_index = random.randint(0, NUMBER_OF_HISTORICAL_COPIES_TO_KEEP-1)
        make_move_historical_for_index = functools.partial(make_move_historical, opponent_index)

        # randomize if going first or second
        if bool(random.getrandbits(1)):
            reward = game_spec.play_game(make_training_move, make_move_historical_for_index)
        else:
            reward = -game_spec.play_game(make_move_historical_for_index, make_training_move)

        results.append(reward)

        last_game_length = len(mini_batch_board_states) - len(mini_batch_rewards)

        # we scale here so winning quickly is better winning slowly and loosing slowly better than loosing quick
        reward /= float(last_game_length)

        mini_batch_rewards += ([reward] * last_game_length)

        episode_number += 1

        if episode_number % MINI_BATCH_SIZE == 0:
            normalized_rewards = mini_batch_rewards - np.mean(mini_batch_rewards)
            normalized_rewards /= np.std(normalized_rewards)

            session.run(train_step, feed_dict={input_layer: mini_batch_board_states,
                                               reward_placeholder: normalized_rewards,
                                               actual_move_placeholder: mini_batch_moves})

            # clear batches
            del mini_batch_board_states[:]
            del mini_batch_moves[:]
            del mini_batch_rewards[:]

        if episode_number % PRINT_RESULTS_EVERY_X == 0:
            print("episode: %s average result: %s" % (episode_number, np.mean(results)))

        if episode_number % SAVE_HISTORICAL_NETWORK_EVERY == 0:
            print("saving historical network %s", current_historical_index)
            save_network(session, variables, BASE_HISTORICAL_NETWORK_PATH + str(current_historical_index) + '.p')
            load_network(session, historical_networks[current_historical_index][2],
                         BASE_HISTORICAL_NETWORK_PATH + str(current_historical_index) + '.p')

            # also save to the main network file
            save_network(session, variables, STARTING_NETWORK_WEIGHTS)

            current_historical_index += 1
            current_historical_index %= NUMBER_OF_HISTORICAL_COPIES_TO_KEEP

    # save our final weights
    save_network(session, variables, STARTING_NETWORK_WEIGHTS)

print("completed")
