from techniques.min_max import min_max_alpha_beta
from techniques.train_policy_gradient import train_policy_gradients
from tic_tac_toe_5_4.network import tic_tac_toe_5_4_game_spec, create_convolutional_network


def min_max_move_func(board_state, side):
    return min_max_alpha_beta(tic_tac_toe_5_4_game_spec, board_state, side, 3)[1]


train_policy_gradients(tic_tac_toe_5_4_game_spec, create_convolutional_network,
                       'convolutional_net_5_4_l_c_4_f_1_other_after.p',
                       opponent_func=min_max_move_func,
                       save_network_file_path='convolutional_net_5_4_l_c_4_f_1_other_after_vs_depth_3.p',
                       number_of_games=5000,
                       print_results_every=100)