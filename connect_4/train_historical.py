from connect_4.network import create_convolutional_network, connect_4_game_spec
from techniques.train_policy_gradient_historic import train_policy_gradients_vs_historic


train_policy_gradients_vs_historic(connect_4_game_spec, create_convolutional_network,
                                   'convolutional_net_5_4_l_c_4_f_1_other_after_1.p',
                                   save_network_file_path='convolutional_net_5_4_l_c_4_f_1_other_after_2.p',
                                   number_of_games=50000,
                                   print_results_every=500,
                                   save_historic_every=8000)