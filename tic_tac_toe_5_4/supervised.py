import pickle

import numpy as np

from techniques.train_supervised import train_supervised
from tic_tac_toe_5_4.network import tic_tac_toe_5_4_game_spec, create_convolutional_network

with open("position_tic_tac_toe_5_4_min_max_depth_6", 'rb') as f:
    positions = pickle.load(f)

# for now we need to reshape input for convolutions and one hot the move responses
# this is the kind of stuff I need to clean up in overall design
for i in range(len(positions)):
    one_hot = np.zeros(tic_tac_toe_5_4_game_spec.outputs())
    np.put(one_hot, tic_tac_toe_5_4_game_spec.tuple_move_to_flat(positions[i][1]), 1)
    positions[i] = np.array(positions[i][0]).reshape(tic_tac_toe_5_4_game_spec.board_dimensions()[0],
                                                     tic_tac_toe_5_4_game_spec.board_dimensions()[0],
                                                     1), one_hot

train_supervised(tic_tac_toe_5_4_game_spec, create_convolutional_network, 'convolutional_net_5_4_l_c_4_f_1_other.p',
                 positions, regularization_coefficent=1e-4)