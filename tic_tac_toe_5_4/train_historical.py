from techniques.train_policy_gradient_historic import train_policy_gradients_vs_historic
from tic_tac_toe_5_4.network import tic_tac_toe_5_4_game_spec, create_convolutional_network

train_policy_gradients_vs_historic(tic_tac_toe_5_4_game_spec, create_convolutional_network,
                                   'convolutional_net_5_4_l_c_4_f_1_other_after_1.p',
                                   save_network_file_path='convolutional_net_5_4_l_c_4_f_1_other_after_2.p',
                                   number_of_games=50000,
                                   print_results_every=500,
                                   save_historic_every=8000)