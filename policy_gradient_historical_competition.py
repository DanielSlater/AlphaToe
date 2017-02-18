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

from common.network_helpers import create_network, load_network, get_stochastic_network_move, \
    save_network
from games.tic_tac_toe import TicTacToeGameSpec
from techniques.train_policy_gradient_historic import train_policy_gradients_vs_historic

HIDDEN_NODES = (100, 100, 100)
SAVE_HISTORICAL_NETWORK_EVERY = 10000
game_spec = TicTacToeGameSpec()

create_network_func = functools.partial(create_network, game_spec.board_squares(), HIDDEN_NODES)

train_policy_gradients_vs_historic(game_spec, create_network_func,
                                   'train_vs_historical.p',
                                   save_historic_every=SAVE_HISTORICAL_NETWORK_EVERY)
