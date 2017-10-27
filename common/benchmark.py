"""
Plays games against a variety of algorithms to see how good a network is
"""
import functools

import tensorflow as tf

from common.network_helpers import load_network, get_deterministic_network_move
from techniques.min_max import min_max_alpha_beta
from techniques.monte_carlo import monte_carlo_tree_search_uct


def benchmark(game_spec, network_file_path, create_network_func, log_games=False, games_vs_random=500):
    """Plays games against a variety of algorithms to see how good a network is. Results are currently just
    printed to std out

    Args:
        game_spec (games.base_game_spec.BaseGameSpec): The game we are playing
        create_network_func (->(input_layer : tf.placeholder, output_layer : tf.placeholder, variables : [tf.Variable])):
            Method that creates the network we will train.
        network_file_path (str): path to the file with weights we want to load for this network
        log_games (bool): If True print all positions from all games played
        games_vs_random (int): Number of games to play vs random opponents
    """
    input_layer, output_layer, variables = create_network_func()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        load_network(session, variables, network_file_path)

        def make_move(board_state, side):
            move = get_deterministic_network_move(session, input_layer, output_layer, board_state, side,
                                                  valid_only=True, game_spec=game_spec)
            return game_spec.flat_move_to_tuple(move.argmax())

        def min_max_move_func(board_state, side, depth):
            return min_max_alpha_beta(game_spec, board_state, side, depth)[1]

        def monte_carlo_move_func(board_state, side):
            return monte_carlo_tree_search_uct(game_spec, board_state, side, 100000)[1]

        results = []
        for _ in range(int(games_vs_random / 2)):
            result = game_spec.play_game(make_move,
                                         game_spec.get_random_player_func(),
                                         log=log_games)
            results.append(result)
            result = game_spec.play_game(
                game_spec.get_random_player_func(),
                make_move, log=log_games)
            results.append(-result)

        print("*** results vs random = %s" % (sum(results),))

        results = []
        for _ in range(1):
            result = game_spec.play_game(make_move,
                                         functools.partial(min_max_move_func, depth=2), log=log_games)
            results.append(result)
            result = game_spec.play_game(functools.partial(min_max_move_func, depth=2),
                                         make_move, log=log_games)
            results.append(-result)

        print("*** results vs min max depth 2 = %s" % (sum(results),))

        results = []
        for _ in range(1):
            result = game_spec.play_game(make_move,
                                         functools.partial(min_max_move_func, depth=4), log=log_games)
            results.append(result)
            result = game_spec.play_game(functools.partial(min_max_move_func, depth=4),
                                         make_move, log=log_games)
            results.append(-result)

        print("*** results vs min max depth 4 = %s" % (sum(results),))

        results = []
        for _ in range(1):
            result = game_spec.play_game(make_move,
                                         functools.partial(min_max_move_func, depth=6), log=log_games)
            results.append(result)
            result = game_spec.play_game(functools.partial(min_max_move_func, depth=6),
                                         make_move, log=log_games)
            results.append(-result)

        print("*** results vs min max depth 6 = %s" % (sum(results),))

        results = []
        for _ in range(1):
            result = game_spec.play_game(make_move,
                                         functools.partial(min_max_move_func, depth=8), log=log_games)
            results.append(result)
            result = game_spec.play_game(functools.partial(min_max_move_func,
                                                           make_move, depth=8), log=log_games)
            results.append(-result)

        print("*** results vs min max depth 8 = %s" % (sum(results),))

        results = []
        for _ in range(1):
            result = game_spec.play_game(make_move,
                                         monte_carlo_move_func, log=log_games)
            results.append(result)
            result = game_spec.play_game(monte_carlo_move_func,
                                         make_move, log=log_games)
            results.append(-result)

        print("*** results vs monte carlo uct 100000 = %s" % (sum(results),))
