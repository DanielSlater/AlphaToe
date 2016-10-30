"""
Builds and trains a neural network that uses policy gradients to learn to play Tic-Tac-Toe.

The input to the network is a vector with a number for each space on the board. If the space has one of the networks
pieces then the input vector has the value 1. -1 for the opponents space and 0 for no piece.

The output of the network is a also of the size of the board with each number learning the probability that a move in
that space is the best move.

The network plays successive games randomly alternating between going first and second against an opponent that makes
moves by randomly selecting a free space. The neural network does NOT initially have any way of knowing what is or is not
a valid move, so initially it must learn the rules of the game.

I have trained this version with success at 3x3 tic tac toe until it has a success rate in the region of 75% this maybe
as good as it can do, because 3x3 tic-tac-toe is a theoretical draw, so the random opponent will often get lucky and
force a draw.
"""
import collections
import os
import random

import numpy as np
import tensorflow as tf

from common.network_helpers import create_network, load_network, get_stochastic_network_move, save_network
from games.tic_tac_toe_x import TicTacToeXGameSpec

HIDDEN_NODES = (200, 160, 120, 80)
BATCH_SIZE = 100  # every how many games to do a parameter update?
LEARN_RATE = 1e-5
PRINT_RESULTS_EVERY_X = 1000  # every how many games to print the results
NETWORK_FILE_PATH = 'current_network.p'  # path to save the network to
NUMBER_OF_GAMES_TO_RUN = 100000

# to play a different game change this to another spec, e.g TicTacToeXGameSpec or ConnectXGameSpec
game_spec = TicTacToeXGameSpec(5, 4) #TicTacToeGameSpec()

reward_placeholder = tf.placeholder("float", shape=(None,))
actual_move_placeholder = tf.placeholder("float", shape=(None, game_spec.outputs()))

input_layer, output_layer, variables = create_network(game_spec.board_squares(), HIDDEN_NODES,
                                                      output_nodes=game_spec.outputs())

policy_gradient = tf.reduce_sum(tf.reshape(reward_placeholder, (-1, 1)) * actual_move_placeholder * output_layer)
train_step = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(-tf.log(policy_gradient))

with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    if os.path.isfile(NETWORK_FILE_PATH):
        print("loading pre-existing network")
        load_network(session, variables, NETWORK_FILE_PATH)

    mini_batch_board_states, mini_batch_moves, mini_batch_rewards = [], [], []
    results = collections.deque(maxlen=PRINT_RESULTS_EVERY_X)


    def make_training_move(board_state, side):
        mini_batch_board_states.append(np.ravel(board_state) * side)
        move = get_stochastic_network_move(session, input_layer, output_layer, board_state, side)
        mini_batch_moves.append(move)
        return game_spec.flat_move_to_tuple(move.argmax())


    for episode_number in range(1, NUMBER_OF_GAMES_TO_RUN):
        # randomize if going first or second
        if bool(random.getrandbits(1)):
            reward = game_spec.play_game(make_training_move, game_spec.get_random_player_func())
        else:
            reward = -game_spec.play_game(game_spec.get_random_player_func(), make_training_move)

        results.append(reward)

        last_game_length = len(mini_batch_board_states) - len(mini_batch_rewards)

        # we scale here so winning quickly is better winning slowly and loosing slowly better than loosing quick
        reward /= float(last_game_length)

        mini_batch_rewards += ([reward] * last_game_length)

        if episode_number % BATCH_SIZE == 0:
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
            print("episode: %s win_rate: %s" % (episode_number, 0.5 + sum(results) / (PRINT_RESULTS_EVERY_X * 2.)))
            save_network(session, variables, NETWORK_FILE_PATH)

    save_network(session, variables, NETWORK_FILE_PATH)
