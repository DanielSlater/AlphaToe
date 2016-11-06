import pickle

import numpy as np

from techniques.train_supervised import train_supervised
from connect_4.network import connect_4_game_spec, create_convolutional_network

with open("position_connect_4_min_max_depth_6", 'rb') as f:
    positions = pickle.load(f)

# for now we need to reshape input for convolutions and one hot the move responses
# this is the kind of stuff I need to clean up in overall design
for i in range(len(positions)):
    one_hot = np.zeros(connect_4_game_spec.outputs())
    np.put(one_hot, positions[i][1], 1)
    positions[i] = np.array(positions[i][0]).reshape(connect_4_game_spec.board_dimensions()[0],
                                                     connect_4_game_spec.board_dimensions()[1],
                                                     1), one_hot

# for convolutional_layers in [3, 4, 5, 6]:
#     for convolutional_channels in [48, 64, 80, 96]:

train_supervised(connect_4_game_spec, create_convolutional_network, 'convolutional_net_l_c_5_f_1_other.p',
                 positions,
                 regularization_coefficent=1e-3,
                 learn_rate=5e-5)