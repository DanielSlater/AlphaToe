import functools
from unittest import TestCase

from common.base_game_spec import BaseGameSpec
from common.network_helpers import create_network
from games.tic_tac_toe import TicTacToeGameSpec
from games.tic_tac_toe_x import TicTacToeXGameSpec
from techniques.train_policy_gradient import train_policy_gradients


class _VerySimpleGameSpec(BaseGameSpec):
    def new_board(self):
        return [0, 0]

    def apply_move(self, board_state, move, side):
        board_state[move] = side
        return board_state

    def has_winner(self, board_state):
        return board_state[0]

    def __init__(self):
        pass

    def available_moves(self, board_state):
        return [i for i, x in enumerate(board_state) if x == 0]

    def board_dimensions(self):
        return 2,


class TestTrainPolicyGradient(TestCase):
    def test_learn_simple_game(self):
        game_spec = _VerySimpleGameSpec()
        create_model_func = functools.partial(create_network, 2, (4,))
        variables, win_rate = train_policy_gradients(game_spec, create_model_func, None,
                                                     learn_rate=0.1,
                                                     number_of_games=1000, print_results_every=100,
                                                     batch_size=20,
                                                     randomize_first_player=False)
        self.assertGreater(win_rate, 0.9)

    def test_tic_tac_toe(self):
        game_spec = TicTacToeGameSpec()
        create_model_func = functools.partial(create_network, game_spec.board_squares(), (100, 100, 100,))
        variables, win_rate = train_policy_gradients(game_spec, create_model_func, None,
                                                     learn_rate=1e-4,
                                                     number_of_games=60000,
                                                     print_results_every=1000,
                                                     batch_size=100,
                                                     randomize_first_player=False)
        self.assertGreater(win_rate, 0.4)
